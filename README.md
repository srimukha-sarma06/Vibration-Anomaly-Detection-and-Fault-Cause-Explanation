# 🛠️ Edge-Light Bearing Fault Diagnosis System

A lightweight predictive maintenance system for detecting faults in
rotating machinery using a hybrid **machine learning + signal analysis**
approach.

The system processes vibration signals and outputs:

-   Machine Health Index
-   Fault Location
-   Fault Severity

Designed specifically for resource‑constrained edge devices such as
Raspberry Pi‑class hardware.

------------------------------------------------------------------------

## Motivation

Industrial monitoring systems are often cloud‑dependent and
computationally heavy.\
This project demonstrates that reliable fault diagnostics can run
locally on low‑power hardware by combining:

-   Statistical anomaly detection
-   Compact neural networks
-   Interpretable condition scoring

The goal is deployability and reliability rather than raw benchmark
accuracy.

------------------------------------------------------------------------

## Features

### Hybrid Detection

-   Isolation Forest --- statistical vibration anomaly detection
-   Convolutional Autoencoder --- spectral anomaly detection

### Condition Interpretation

-   Normalized multi‑model fusion
-   Fault region inference from calibrated health index
-   Progressive severity estimation

### Lightweight Design

-   Autoencoder model ≈ 250 KB
-   Fully offline operation
-   Deterministic outputs suitable for live demos

------------------------------------------------------------------------

## Example Output

Health Index : 8.92
Fault : Outer Race
Severity : High Fault

------------------------------------------------------------------------

## Installation

pip install -r requirements.txt

------------------------------------------------------------------------

## Running

Place .mat vibration files inside:

test_data/

Then run:

python main.py

------------------------------------------------------------------------

## System Pipeline

1.  Signal Processing
    -   Window segmentation
    -   Feature extraction
    -   Spectrogram generation
2.  Dual Anomaly Detection
    -   Isolation Forest evaluates statistical deviation
    -   Autoencoder measures reconstruction error
3.  Health Fusion
    -   Normalized combination into a single interpretable health index
4.  Diagnosis
    -   Health index determines fault location and severity

------------------------------------------------------------------------

## Health Index Interpretation

Score \< 8.4 → Healthy\
8.4 -- 8.85 → Inner Race Fault\
\> 8.85 → Outer Race Fault

Severity depends on progression within the region.

------------------------------------------------------------------------

## Runtime Requirements

Python 3.9+\
CPU only --- no GPU required\
Offline capable

Model sizes: - Autoencoder ≈ 250 KB - Isolation Forest + scaler ~ 4 MB

------------------------------------------------------------------------

## Purpose

Demonstrates that predictive maintenance systems can be lightweight,
interpretable, and edge‑deployable instead of cloud‑dependent.
