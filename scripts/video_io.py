import cv2
import numpy as np


def _sample_indices(num_available, frame_count, random_sample, rng):
    """Sceglie gli indici dei frame da estrarre.

    - random_sample=False (val/test, anteprima): campionamento uniforme
      deterministico, come prima (np.linspace).
    - random_sample=True (training): campionamento stile TSN. Il video viene
      diviso in `frame_count` segmenti e da ciascuno si pesca un frame a caso.
      Cosi' ogni epoca il modello vede frame leggermente diversi dello stesso
      clip (augmentation temporale) ma con copertura uniforme del video.
    """
    if num_available <= 0:
        return np.zeros(frame_count, dtype=np.int32)

    if not random_sample:
        return np.linspace(0, num_available - 1, frame_count).astype(np.int32)

    rng = rng if rng is not None else np.random
    bounds = np.linspace(0, num_available, frame_count + 1)
    indices = np.empty(frame_count, dtype=np.int32)
    for i in range(frame_count):
        lo = int(np.floor(bounds[i]))
        hi = int(np.ceil(bounds[i + 1])) - 1
        hi = min(max(hi, lo), num_available - 1)
        indices[i] = rng.randint(lo, hi + 1)
    return indices


def decode_video(path, frame_count=16, image_size=160, random_sample=False, rng=None):
    path = str(path)
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        print(f'[WARN] impossibile aprire: {path}')
        return np.zeros((frame_count, image_size, image_size, 3), dtype=np.float32)

    # Leggi tutti i frame disponibili senza fidarti di CAP_PROP_FRAME_COUNT
    all_frames = []
    while True:
        ok, frame = capture.read()
        if not ok or frame is None:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_AREA)
        all_frames.append(frame.astype(np.float32) / 255.0)
    capture.release()

    if not all_frames:
        print(f'[WARN] nessun frame letto: {path}')
        return np.zeros((frame_count, image_size, image_size, 3), dtype=np.float32)

    # Campiona frame_count frame (uniforme in val/test, stocastico in training)
    indices = _sample_indices(len(all_frames), frame_count, random_sample, rng)
    frames = [all_frames[i] for i in indices]
    return np.stack(frames, axis=0).astype(np.float32)
