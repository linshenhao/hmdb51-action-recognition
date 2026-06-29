import tensorflow as tf


def _weights_value(weights):
    if weights is None:
        return None
    val = str(weights).strip().lower()
    return None if val in {'', 'none', 'null', 'false'} else val


def _build_conv3d(num_classes, frame_count, image_size, dropout):
    inputs = tf.keras.Input(shape=(frame_count, image_size, image_size, 3), name='video')

    x = tf.keras.layers.Conv3D(32, (3, 5, 5), padding='same', use_bias=False, name='stem_conv')(inputs)
    x = tf.keras.layers.BatchNormalization(name='stem_bn')(x)
    x = tf.keras.layers.Activation('relu', name='stem_relu')(x)
    x = tf.keras.layers.MaxPool3D(pool_size=(1, 2, 2), name='stem_pool')(x)

    for i, filters in enumerate([64, 96, 128], start=1):
        n = f'block{i}'
        x = tf.keras.layers.Conv3D(filters, 3, padding='same', use_bias=False, name=f'{n}_conv1')(x)
        x = tf.keras.layers.BatchNormalization(name=f'{n}_bn1')(x)
        x = tf.keras.layers.Activation('relu', name=f'{n}_relu1')(x)
        x = tf.keras.layers.Conv3D(filters, 3, padding='same', use_bias=False, name=f'{n}_conv2')(x)
        x = tf.keras.layers.BatchNormalization(name=f'{n}_bn2')(x)
        x = tf.keras.layers.Activation('relu', name=f'{n}_relu2')(x)
        x = tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2), name=f'{n}_pool')(x)

    x = tf.keras.layers.GlobalAveragePooling3D(name='global_pool')(x)
    x = tf.keras.layers.Dropout(dropout, name='dropout')(x)
    x = tf.keras.layers.Dense(256, activation='relu', name='dense_features')(x)
    x = tf.keras.layers.Dropout(dropout, name='classifier_dropout')(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax', name='class_probs')(x)
    return tf.keras.Model(inputs, outputs, name='conv3d')


def _build_mobilenet(num_classes, frame_count, image_size, temporal_head, weights, backbone_trainable, dropout):
    inputs = tf.keras.Input(shape=(frame_count, image_size, image_size, 3), name='video')
    x = tf.keras.layers.Rescaling(scale=2.0, offset=-1.0, name='mobilenet_preprocess')(inputs)

    wanted = _weights_value(weights)
    try:
        backbone = tf.keras.applications.MobileNetV2(
            input_shape=(image_size, image_size, 3), include_top=False,
            weights=wanted, pooling='avg',
        )
        if wanted == 'imagenet':
            print('[OK] MobileNetV2: pesi ImageNet caricati.')
    except Exception as exc:
        if wanted == 'imagenet':
            # Con backbone casuale il modello resta al livello del caso
            # (1/num_classes), quindi falliamo esplicitamente invece di
            # proseguire con pesi random. NON assumiamo che la causa sia il
            # download: riportiamo l'errore REALE cosi' e' diagnosticabile
            # (es. h5py mancante, versione TF incompatibile, cache corrotta).
            raise RuntimeError(
                'Impossibile costruire MobileNetV2 con pesi ImageNet. '
                f'Causa reale: {type(exc).__name__}: {exc}. '
                'Senza pesi pre-addestrati il modello non converge. '
                'Verifica la causa qui sopra (connessione, h5py installato, '
                'versione di TensorFlow), oppure imposta WEIGHTS="none" '
                'consapevolmente per addestrare da zero.'
            ) from exc
        backbone = tf.keras.applications.MobileNetV2(
            input_shape=(image_size, image_size, 3), include_top=False,
            weights=None, pooling='avg',
        )
    backbone.trainable = backbone_trainable

    x = tf.keras.layers.TimeDistributed(backbone, name='frame_encoder')(x)
    x = tf.keras.layers.LayerNormalization(name='temporal_norm')(x)
    if temporal_head == 'gru':
        x = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(128, dropout=dropout), name='bidirectional_gru')(x)
    else:
        x = tf.keras.layers.GlobalAveragePooling1D(name='temporal_average')(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax', name='class_probs')(x)
    return tf.keras.Model(inputs, outputs, name=f'mobilenet_{temporal_head}')


def smoothed_sparse_ce(num_classes, label_smoothing=0.0):
    """Sparse categorical cross-entropy con label smoothing opzionale.

    `SparseCategoricalCrossentropy` non supporta `label_smoothing`, quindi quando
    serve convertiamo le label intere in one-hot e usiamo `CategoricalCrossentropy`.
    Le metriche 'sparse' (accuracy, top-5) restano valide perche' continuano a
    ricevere le label intere: solo la loss lavora internamente in one-hot.

    Con label_smoothing=0 ritorna la stringa standard (nessun overhead).
    """
    if not label_smoothing:
        return 'sparse_categorical_crossentropy'
    cce = tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing)

    def loss_fn(y_true, y_pred):
        y_true = tf.one_hot(tf.cast(tf.reshape(y_true, [-1]), tf.int32), num_classes)
        return cce(y_true, y_pred)

    loss_fn.__name__ = 'sparse_ce_label_smoothing'
    return loss_fn


def unfreeze_backbone(model, from_block=None, freeze_bn=True):
    """Sblocca il backbone MobileNetV2 (dentro il TimeDistributed) per il fine-tuning.

    - from_block=None: sblocca tutto il backbone.
    - from_block='block_11': sblocca solo dai layer di quel blocco in poi
      (fine-tuning PARZIALE: piu' stabile e meno incline all'overfitting, perche'
      i primi blocchi - feature generiche di bordi/texture - restano congelati).
    - freeze_bn=True: tiene le BatchNormalization in inference. Con batch piccoli
      aggiornarne media/varianza destabilizza le feature ImageNet.

    Ritorna (n_layer_addestrabili, n_batchnorm_congelate).
    """
    backbone = model.get_layer('frame_encoder').layer
    backbone.trainable = True
    reached = from_block is None
    n_train, n_bn = 0, 0
    for layer in backbone.layers:
        if from_block is not None and layer.name.startswith(from_block + '_'):
            reached = True
        if not reached:
            layer.trainable = False
            continue
        if freeze_bn and isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
            n_bn += 1
        else:
            layer.trainable = True
            n_train += 1
    return n_train, n_bn


def build_model(model_name, num_classes, frame_count, image_size,
                weights='imagenet', backbone_trainable=False, dropout=0.35):
    if model_name == 'conv3d':
        return _build_conv3d(num_classes, frame_count, image_size, dropout)
    if model_name in ('mobilenet_gru', 'mobilenet_avg'):
        head = 'gru' if model_name == 'mobilenet_gru' else 'avg'
        return _build_mobilenet(num_classes, frame_count, image_size, head, weights, backbone_trainable, dropout)
    raise ValueError(f'Modello sconosciuto: {model_name}')
