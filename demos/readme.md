chroma\_stft([y, sr, S, norm, n\_fft, …]) | Compute a chromagram from a waveform or power spectrogram.
:-----:|:-----:
chroma\_cqt([y, sr, C, hop\_length, fmin, …]) |Constant-Q chromagram
chroma\_cens([y, sr, C, hop\_length, fmin, …]) |Computes the chroma variant “Chroma Energy Normalized” (CENS), following [R674badebce0d-1].
melspectrogram([y, sr, S, n\_fft, …]) |Compute a mel-scaled spectrogram.
mfcc([y, sr, S, n\_mfcc, dct\_type, norm]) |Mel-frequency cepstral coefficients (MFCCs)
rmse([y, S, frame\_length, hop\_length, …]) |Compute root-mean-square (RMS) energy for each frame, either from the audio samples y or from a spectrogram S.
spectral\_centroid([y, sr, S, n\_fft, …]) |Compute the spectral centroid.
spectral\_bandwidth([y, sr, S, n\_fft, …]) |Compute p’th-order spectral bandwidth:
spectral\_contrast([y, sr, S, n\_fft, …]) |Compute spectral contrast [R6ffcc01153df-1]
spectral\_flatness([y, S, n\_fft, hop\_length, …]) |Compute spectral flatness
spectral\_rolloff([y, sr, S, n\_fft, …]) |Compute roll-off frequency.
poly\_features([y, sr, S, n\_fft, hop\_length, …]) |Get coefficients of fitting an nth-order polynomial to the columns of a spectrogram.
tonnetz([y, sr, chroma]) |Computes the tonal centroid features (tonnetz), following the method of [Recf246e5a035-1].
zero\_crossing\_rate(y[, frame\_length, …]) |Compute the zero-crossing rate of an audio time series.
