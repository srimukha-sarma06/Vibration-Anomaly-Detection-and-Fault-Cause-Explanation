import numpy as np
from scipy.io import loadmat

def get_signal(mat_path):
    data = loadmat(mat_path)

    fs = None
    signal = None

    for key, val in data.items():

        # skip matlab metadata
        if key.startswith("__"):
            continue

        arr = np.asarray(val)

        # sampling rate (usually scalar)
        if arr.size == 1 and np.issubdtype(arr.dtype, np.number):
            fs = float(arr.squeeze())
            continue

        # vibration signal → long numeric vector
        if arr.ndim in (1,2) and max(arr.shape) > 100:
            signal = arr.squeeze().astype(np.float32)

    if signal is None:
        raise ValueError("No vibration signal found in mat file")

    if fs is None:
        fs = 12000  # fallback default

    return signal, fs
