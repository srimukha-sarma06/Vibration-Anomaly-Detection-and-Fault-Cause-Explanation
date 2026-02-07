import os
import joblib
import tensorflow as tf
import numpy as np
import cv2
from scipy.signal import spectrogram
from scipy.io import loadmat
from get_signal import get_signal
from make_image_data import spectrogram_batch, spec_from_window, windows

folder = "/normal_signals"   # change if inside sample_data

healthy_spec_files = [
    os.path.join(folder, f)
    for f in os.listdir(folder)
    if f.startswith("Normal") and f.endswith(".mat")
]

normal_errors = []
conv_model = tf.keras.models.load_model('models/Conv_Autoencoder_16x16.keras')

for f in healthy_spec_files:

    signal, fs = get_signal(f)      # RAW SIGNAL, not S_gray
    imgs = spectrogram_batch(signal, fs)   # (N,32,32,1)

    recon = conv_model.predict(imgs, verbose=0)

    err = ((imgs - recon)**2).mean(axis=(1,2,3))  # error per patch
    normal_errors.extend(err)   # IMPORTANT: extend, not append

baseline_mean_ae = np.mean(normal_errors)
baseline_std_ae  = np.std(normal_errors)

print("AE baseline mean:", baseline_mean_ae)
print("AE baseline std :", baseline_std_ae)

joblib.dump((baseline_mean_ae, baseline_std_ae), "baseline_ae.joblib")
