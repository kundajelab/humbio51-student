from scipy.signal import fftconvolve
import numpy as np
from math import ceil 
def scan_sequence(sequence,pssm):
    convolution_product=fftconvolve(sequence[::-1,::-1],pssm,mode="same")[::-1,::-1][1]
    #get the starting position of the motif along the sequence
    starting_pos=int(ceil(np.argmax(convolution_product)-pssm.shape[1]/2))
    return convolution_product,starting_pos
