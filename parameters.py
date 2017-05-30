fs = 48000
framesize_t = 0.025     # in second
hopsize_t   = 0.010

framesize   = int(round(framesize_t*fs))
hopsize     = int(round(hopsize_t*fs))

# MFCC params
highFrequencyBound = fs/2 if fs/2<11000 else 11000