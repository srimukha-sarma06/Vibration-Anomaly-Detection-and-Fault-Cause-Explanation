import numpy as np

def severity(z):
  if z < 8.5:
    return "Normal"
  elif z < 8.8:
    return "Early Fault"
  else:
     return "High Fault"

  
def normalize(z, scale=2.0):
    # maps 0→0 , 2→0.63 , 5→0.92
    return 1 - np.exp(-z/scale)
  
def combined_health_state(z_ae, z_iso):

    # convert both detectors to same meaning scale
    h_ae  = normalize(z_ae)
    h_iso = normalize(z_iso)

    # fuse evidence (IF slightly more trusted)
    combined = 0.6*h_iso + 0.4*h_ae

    # convert back to readable 0–10 health index
    health = combined * 10

    state = severity(health)

    return health, state