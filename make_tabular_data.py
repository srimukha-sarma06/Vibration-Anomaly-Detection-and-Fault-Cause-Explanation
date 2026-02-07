from scipy.stats import skew, kurtosis
import numpy as np
from windows import windows

def feature_vector(w):

    sd = np.std(w)
    rms = np.sqrt(np.mean(w**2))
    sk  = skew(w)
    ku  = kurtosis(w)

    crest = np.max(np.abs(w)) / (rms + 1e-8)

    mean_abs = np.mean(np.abs(w))
    form = rms / (mean_abs + 1e-8)

    return np.array([sd, rms, sk, ku, crest, form], dtype=np.float32)

def features_from_signal(sig):
    feats = [feature_vector(w) for w in windows(sig)]
    return np.vstack(feats)
