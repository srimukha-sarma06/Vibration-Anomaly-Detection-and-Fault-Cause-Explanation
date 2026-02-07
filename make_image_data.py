from scipy.signal import spectrogram
import numpy as np
import cv2
from windows import windows

def spec_from_window(w, fs):

    # avoid divide-by-zero but keep info
    w = w.astype(np.float32)
    w = (w - np.mean(w)) / (np.std(w) + 1e-8)

    # spectrogram
    f, t, S = spectrogram(w, fs=fs, nperseg=256, noverlap=128)

    # log power
    S = np.log(S + 1e-12)

    # normalize PER IMAGE (critical for AE)
    S = (S - S.mean()) / (S.std() + 1e-8)

    # resize to model input
    S = cv2.resize(S, (32,32), interpolation=cv2.INTER_AREA)

    return S.astype(np.float32)

def spectrogram_batch(signal, fs):

    specs = []
    for w in windows(signal):
        img = spec_from_window(w, fs)

        # discard NaN windows
        if not np.isfinite(img).all():
            continue

        specs.append(img)

    specs = np.array(specs, dtype=np.float32)

    # (N,32,32,1) for keras
    specs = specs[..., np.newaxis]

    return specs
