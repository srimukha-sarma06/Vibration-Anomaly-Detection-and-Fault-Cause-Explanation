import tensorflow as tf
import numpy as np
import joblib
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
from scipy.signal import spectrogram
from scipy.io import loadmat
import cv2
from machine_health import severity, combined_health_state, normalize
from make_image_data import spec_from_window, spectrogram_batch
from make_tabular_data import windows, feature_vector, features_from_signal
from get_signal import get_signal
from fault_detection import locate_from_health

#Loading the models
ae = tf.keras.models.load_model('models/Conv_Autoencoder_16x16.keras')
iso = joblib.load('models/bearing_iforest_final.joblib')
scaler = joblib.load('models/scaler.joblib')


def Detect(path):
    #Getting the baseline metrics for calculating the machine health
    baseline_mean_iso, baseline_std_iso = joblib.load('baseline_metrics/iforest_baseline.joblib')
    baseline_mean_ae, baseline_std_ae = joblib.load('baseline_metrics/baseline_ae.joblib')

    #Obtaining the signal from .mat file
    signal, fs = get_signal(path)

    #Obtaining the image and predicting using the autoencoder
    img = spectrogram_batch(signal, fs)

    if len(img) == 0:
        return {"health_value":0, "severity":"Signal Error", "fault_location":"Unknown"}

    recon = ae.predict(img, verbose=0)


    errors = np.mean((img - recon)**2, axis=(1,2,3))

    error = np.abs((errors - baseline_mean_ae) / baseline_std_ae)
    machine_health_ae = float(np.median(error))
    print(machine_health_ae)

    #Obtaining the tabular data and predicting using the isolation forest
    X = features_from_signal(signal) 

    if not np.isfinite(X).all():
        return 0, "Sensor Fault"

    X_scaled = scaler.transform(X)
    scores = iso.decision_function(X_scaled)

    error = np.abs((baseline_mean_iso - scores) / baseline_std_iso)
    machine_health_iso = float(np.median(error))
    print(machine_health_iso)

    #Normalizing the scores
    combined_health, state = combined_health_state(machine_health_ae, machine_health_iso)

    fault_type= locate_from_health(combined_health)
    if state != "Normal":
        reason = "Detected periodic impacts consistent with " + fault_type
    else:
        reason = "Satisfactory Health Score!"

    return {
        "health_value": combined_health,
        "severity": state,
        "fault_location": fault_type,
        "reason" : reason
    }

if __name__ == '__main__':
    outputs = Detect('test_data/OR021_6_1_239.mat')
    print(outputs)





