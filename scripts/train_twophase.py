"""Training a due fasi (Fase A testa congelata + Fase B fine-tuning) per HMDB51.
Replica la config del notebook HMDB51_TensorFlow_Classifier.ipynb e usa i moduli in scripts/.
"""
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Certificati per eventuali download (pesi gia' in cache, ma per sicurezza)
try:
    import certifi
    os.environ.setdefault('SSL_CERT_FILE', certifi.where())
except Exception:
    pass

import numpy as np
import tensorflow as tf

SCRIPTS_DIR = Path(__file__).resolve().parent      # .../HMDB project/scripts
PROJECT_ROOT = SCRIPTS_DIR.parent                  # .../HMDB project
sys.path.insert(0, str(SCRIPTS_DIR))               # i moduli stanno qui dentro

from data import make_dataset, read_class_names
from models import build_model, smoothed_sparse_ce, unfreeze_backbone
from metrics import evaluate_to_dir

# ---------------- Config (uguale al notebook) ----------------
SEED = 42
SPLIT_ID = 1
MODEL_NAME = 'mobilenet_gru'
WEIGHTS = 'imagenet'
DROPOUT = 0.5
LABEL_SMOOTHING = 0.1   # contro l'overconfidence (val_loss che sale mentre val_acc e' stabile)

FRAMES = 16
IMAGE_SIZE = 160        # dimensione nativa MobileNetV2 (valide: 96/128/160/192/224)
BATCH_SIZE = 8

EPOCHS_HEAD = 20
LR_HEAD = 1e-3
DO_FINETUNE = True
EPOCHS_FINETUNE = 15
LR_FINETUNE = 3e-5
FINETUNE_FROM_BLOCK = 'block_11'   # sblocco PARZIALE: solo dagli ultimi blocchi (None = tutto il backbone)

QUICK_LIMIT_PER_CLASS = None
CACHE_DATASET = False

tf.keras.utils.set_random_seed(SEED)

MANIFEST_DIR = PROJECT_ROOT / 'dataset' / 'manifests'
OUTPUT_DIR = PROJECT_ROOT / 'outputs'
TRAIN_MANIFEST = MANIFEST_DIR / f'split{SPLIT_ID}_train.csv'
VAL_MANIFEST = MANIFEST_DIR / f'split{SPLIT_ID}_val.csv'
TEST_MANIFEST = MANIFEST_DIR / f'split{SPLIT_ID}_test.csv'
CLASSES_PATH = MANIFEST_DIR / 'classes.txt'

class_names = read_class_names(CLASSES_PATH)
print(f'Classi: {len(class_names)}', flush=True)

train_ds, train_count = make_dataset(TRAIN_MANIFEST, FRAMES, IMAGE_SIZE, BATCH_SIZE,
                                     training=True, seed=SEED,
                                     limit_per_class=QUICK_LIMIT_PER_CLASS, cache=CACHE_DATASET)
val_ds, val_count = make_dataset(VAL_MANIFEST, FRAMES, IMAGE_SIZE, BATCH_SIZE,
                                 training=False, seed=SEED,
                                 limit_per_class=QUICK_LIMIT_PER_CLASS, cache=CACHE_DATASET)
print(f'Train: {train_count}  Val: {val_count}', flush=True)

# ---------------- FASE A: backbone congelato ----------------
model = build_model(model_name=MODEL_NAME, num_classes=len(class_names),
                    frame_count=FRAMES, image_size=IMAGE_SIZE, weights=WEIGHTS,
                    backbone_trainable=False, dropout=DROPOUT)
loss_fn = smoothed_sparse_ce(len(class_names), label_smoothing=LABEL_SMOOTHING)
metrics = ['accuracy',
           tf.keras.metrics.SparseTopKCategoricalAccuracy(k=min(5, len(class_names)), name='top5_accuracy')]
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR_HEAD),
              loss=loss_fn, metrics=metrics)

timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
RUN_DIR = OUTPUT_DIR / f'{timestamp}_split{SPLIT_ID}_{MODEL_NAME}_twophase'
RUN_DIR.mkdir(parents=True, exist_ok=False)
print('Run directory:', RUN_DIR, flush=True)

import shutil
shutil.copyfile(CLASSES_PATH, RUN_DIR / 'classes.txt')
with (RUN_DIR / 'notebook_config.json').open('w', encoding='utf-8') as fh:
    json.dump({'split_id': SPLIT_ID, 'model_name': MODEL_NAME, 'weights': WEIGHTS,
               'dropout': DROPOUT, 'label_smoothing': LABEL_SMOOTHING,
               'frames': FRAMES, 'image_size': IMAGE_SIZE,
               'batch_size': BATCH_SIZE, 'epochs_head': EPOCHS_HEAD, 'lr_head': LR_HEAD,
               'do_finetune': DO_FINETUNE, 'epochs_finetune': EPOCHS_FINETUNE,
               'lr_finetune': LR_FINETUNE, 'finetune_from_block': FINETUNE_FROM_BLOCK}, fh, indent=2)


def make_callbacks(history_csv, initial_threshold=None):
    """Tutte le callback monitorano val_accuracy (coerenti). ReduceLROnPlateau non
    segue piu' val_loss, che saliva per overconfidence azzerando il LR troppo presto."""
    return [
        tf.keras.callbacks.ModelCheckpoint(str(RUN_DIR / 'best_model.keras'),
            monitor='val_accuracy', mode='max', save_best_only=True,
            initial_value_threshold=initial_threshold),
        tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', mode='max',
            patience=8, min_delta=1e-3, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', mode='max',
            factor=0.3, patience=4, min_lr=1e-7),
        tf.keras.callbacks.CSVLogger(str(RUN_DIR / history_csv)),
    ]


callbacks = make_callbacks('history.csv')

print('=== FASE A: alleno la testa (backbone congelato) ===', flush=True)
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_HEAD,
                    callbacks=callbacks, verbose=2)

# ---------------- FASE B: fine-tuning ----------------
if DO_FINETUNE:
    print(f'=== FASE B: fine-tuning del backbone da {FINETUNE_FROM_BLOCK} ===', flush=True)
    # Sblocco PARZIALE (solo ultimi blocchi) con BatchNorm in inference: con batch
    # piccoli aggiornarne media/varianza destabilizza le feature ImageNet.
    n_train, n_bn = unfreeze_backbone(model, from_block=FINETUNE_FROM_BLOCK, freeze_bn=True)
    print(f'Backbone: {n_train} layer addestrabili, {n_bn} BatchNorm in inference.', flush=True)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR_FINETUNE),
                  loss=loss_fn, metrics=metrics)
    best_so_far = max(history.history['val_accuracy'])
    history_ft = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_FINETUNE,
                           callbacks=make_callbacks('history_finetune.csv', best_so_far), verbose=2)

# ---------------- Valutazione su test ----------------
print('=== Valutazione su test set ===', flush=True)
best_model = tf.keras.models.load_model(RUN_DIR / 'best_model.keras', compile=False)
test_ds, test_count = make_dataset(TEST_MANIFEST, FRAMES, IMAGE_SIZE, BATCH_SIZE,
                                   training=False, seed=SEED,
                                   limit_per_class=QUICK_LIMIT_PER_CLASS, cache=CACHE_DATASET)
metrics = evaluate_to_dir(best_model, test_ds, class_names, RUN_DIR / 'test_eval')
print('Test records:', test_count, flush=True)
print('METRICHE FINALI:', json.dumps(metrics, indent=2), flush=True)
print('FATTO. Risultati in:', RUN_DIR, flush=True)
