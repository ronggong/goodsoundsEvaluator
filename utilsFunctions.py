import numpy as np

def hz2cents(pitchInHz, tonic=261.626):
    cents = 1200*np.log2(1.0*pitchInHz/tonic)
    return cents