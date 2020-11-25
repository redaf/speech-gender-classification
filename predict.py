import os
import pickle
import sys
import librosa
import numpy as np
import scipy

def mean_frequency(freqs, magn):
    return np.average(a=freqs, weights=magn)

def extract_features_fun(y, sr, fmax=280):
    f0 = librosa.yin(y=y, fmin=1, fmax=fmax, sr=sr, frame_length=512)
    meanfun = np.mean(f0) / 1000
    sd = np.std(f0) / 1000
    iqr = scipy.stats.iqr(f0) / 1000
    return meanfun, sd, iqr

def extract_features_freq(y, sr, fmax=280):
    freqs = librosa.fft_frequencies(sr=sr, n_fft=512)
    stft = librosa.stft(y=y, n_fft=512)

    freqs = freqs[freqs <= fmax]
    stft = stft[0:len(freqs)]
    
    magn = np.abs(stft) + 1e-8

    avg = np.zeros(magn.shape[1])
    for t in range(0, magn.shape[1]):
        avg[t] = mean_frequency(freqs, magn[:, t])

    meanfreq = np.mean(avg) / 1000
    sd = np.std(avg) / 1000
    iqr = scipy.stats.iqr(avg) / 1000
    return meanfreq, sd, iqr



if len(sys.argv) < 2:
    print('Usage: python predict.py path/to/wavfile')
    exit(1)

models_dir = 'models'

model_files = os.listdir(models_dir)

models = {}

for model_file in model_files:
    model_name = os.path.splitext(model_file)[0]
    filename = os.path.join(models_dir, model_file)
    models[model_name] = pickle.load(open(filename, 'rb'))

scaler = models['scaler']

models.pop('scaler')

filename = sys.argv[1]

signal, sample_rate = librosa.load(filename)
# meanfun, _, iqr = extract_features_fun(y=signal, sr=sample_rate)
meanfreq, sd, iqr = extract_features_freq(y=signal, sr=sample_rate)
x = np.array([meanfreq, sd, iqr], ndmin=2)
x = scaler.transform(x)

acc_result = 0

for model_name, model in models.items():
    acc_result += model.predict(x)[0]
    
if acc_result < 2.5:
    print('man')
else:
    print('woman')
