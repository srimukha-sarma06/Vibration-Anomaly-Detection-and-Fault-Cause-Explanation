import numpy as np

def windows(x, win=1024, step=512):

    if len(x) < win:
        pad = win - len(x)
        x = np.pad(x, (0, pad), mode='reflect')

    for i in range(0, len(x)-win+1, step):
        yield x[i:i+win]