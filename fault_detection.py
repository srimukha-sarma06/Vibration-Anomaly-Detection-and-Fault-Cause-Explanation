import numpy as np
from scipy.signal import hilbert
from windows import windows
from collections import Counter

def envelope_spectrum(signal, fs):

    analytic = hilbert(signal)
    envelope = np.abs(analytic)

    spec = np.abs(np.fft.rfft(envelope))
    freqs = np.fft.rfftfreq(len(envelope), 1/fs)

    # remove DC and very low freq drift
    valid = freqs > 5
    return freqs[valid], spec[valid]

def dominant_impact_frequency(freqs, spec):

    # ignore very low freq machine rumble
    mask = freqs > 20
    freqs = freqs[mask]
    spec = spec[mask]

    if len(spec) == 0:
        return 0

    peak_idx = np.argmax(spec)
    return freqs[peak_idx]

def harmonic_pattern(freqs, spec, f0):

    harmonics = []

    for k in range(1, 6):
        target = k * f0
        idx = np.argmin(np.abs(freqs - target))
        harmonics.append(spec[idx])

    harmonics = np.array(harmonics)

    mean_energy = np.mean(harmonics) + 1e-12
    spread = np.std(harmonics) / mean_energy

    return spread, harmonics

def detect_fault(signal, fs, rpm=None):

    freqs, spec = envelope_spectrum(signal, fs)

    if len(spec) < 10:
        return "No localized defect", {"reason": "weak signal"}

    impact_freq = dominant_impact_frequency(freqs, spec)

    if impact_freq == 0:
        return "No localized defect", {"reason": "no peak"}

    spread, harmonics = harmonic_pattern(freqs, spec, impact_freq)

    if spread < 0.25:
        fault = "Outer Race"
    elif spread < 0.55:
        fault = "Inner Race"
    else:
        fault = "Ball"

    # reliability guard
    if np.mean(harmonics) < 3 * np.median(spec):
        fault = "No localized defect"

    info = {
        "impact_frequency": float(impact_freq),
        "harmonic_spread": float(spread)
    }

    return fault, info

def locate_fault(signal, fs, rpm):

    votes = []
    infos = []

    for w in windows(signal):    
        fault, info = detect_fault(w, fs, rpm)
        votes.append(fault)
        infos.append(info)

    # remove "No defect"
    votes = [v for v in votes if v != "No localized defect"]

    if len(votes) == 0:
        return "No localized defect", {}

    # majority vote
    final = Counter(votes).most_common(1)[0][0]

    return final, {"votes": votes}

