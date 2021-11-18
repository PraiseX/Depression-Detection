# from python_speech_features import mfcc
# from python_speech_features import logfbank
# import math
# import numpy as np
# from scikits.talkbox.linpred.levinson_lpc import levinson, acorr_lpc, lpc

# import scipy.io.wavfile as wav

# (rate,sig) = wav.read("file.wav")
# mfcc_feat = mfcc(sig,rate)
# fbank_feat = logfbank(sig,rate)
#
# print(fbank_feat[1:3,:])
#

#lsf implementation https://pyspectrum.readthedocs.io/en/latest/install.html

# def lpc(y, m):
#     "Return m linear predictive coefficients for sequence y using Levinson-Durbin prediction algorithm"
#     #step 1: compute autoregression coefficients R_0, ..., R_m
#     R = [y.dot(y)]
#     if R[0] == 0:
#         return [1] + [0] * (m-2) + [-1]
#     else:
#         for i in range(1, m + 1):
#             r = y[i:].dot(y[:-i])
#             R.append(r)
#         R = np.array(R)
#     #step 2:
#         A = np.array([1, -R[1] / R[0]])
#         E = R[0] + R[1] * A[1]
#         for k in range(1, m):
#             if (E == 0):
#                 E = 10e-17
#             alpha = - A[:k+1].dot(R[k+1:0:-1]) / E
#             A = np.hstack([A,0])
#             A = A + alpha * A[::-1]
#             E *= (1 - alpha**2)
#         return A

import scipy.io
from spafe.utils import vis
from spafe.features.lfcc import lfcc
from spafe.features.lpc import lpc, lpcc
from spafe.features.mfcc import mfcc, imfcc


# init input vars
num_ceps = 13
low_freq = 0
high_freq = 2000
nfilts = 24
nfft = 512
dct_type = 2,
use_energy = False,
lifter = 5
normalize = False


# read wav
# noinspection PyUnresolvedReferences
fs, sig = scipy.io.wavfile.read("D:\MQP\492_AUDIO.wav")

# compute features
lfccs = lfcc(sig=sig,
             fs=fs,
             num_ceps=num_ceps,
             nfilts=nfilts,
             nfft=nfft,
             low_freq=low_freq,
             high_freq=high_freq,
             dct_type=dct_type,
             use_energy=use_energy,
             lifter=lifter,
             normalize=normalize)

# visualize spectogram
vis.spectogram(sig, fs)
# visualize features
vis.visualize_features(lfccs, 'LFCC Index', 'Frame Index')

# compute features
mfccs = mfcc(sig=sig,
             fs=fs,
             num_ceps=num_ceps,
             nfilts=nfilts,
             nfft=nfft,
             low_freq=low_freq,
             high_freq=high_freq,
             dct_type=dct_type,
             use_energy=use_energy,
             lifter=lifter,
             normalize=normalize)

# visualize spectogram
vis.spectogram(sig, fs)
# visualize features
vis.visualize_features(mfccs, 'MFCC Index', 'Frame Index')




# compute features
imfccs = imfcc(sig=sig,
               fs=fs,
               num_ceps=num_ceps,
               nfilts=nfilts,
               nfft=nfft,
               low_freq=low_freq,
               high_freq=high_freq,
               dct_type=dct_type,
               use_energy=use_energy,
               lifter=lifter,
               normalize=normalize)

# visualize features
vis.visualize_features(imfccs, 'IMFCC Index', 'Frame Index')

lpclifter = 0
lpcnormalize = True

# compute lpcs
lpcs = lpc(sig=sig, fs=fs, num_ceps=num_ceps)
# visualize features
vis.visualize_features(lpcs, 'LPC Index', 'Frame Index')


# visualize spectogram
vis.spectogram(sig, fs)
# compute lpccs
lpccs = lpcc(sig=sig, fs=fs, num_ceps=num_ceps, lifter=lpclifter, normalize=lpcnormalize)
# visualize features
vis.visualize_features(lpccs, 'LPCC Index', 'Frame Index')
