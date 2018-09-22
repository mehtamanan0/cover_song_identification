chroma_stft([y, sr, S, norm, n_fft, …]) 	Compute a chromagram from a waveform or power spectrogram.
chroma_cqt([y, sr, C, hop_length, fmin, …]) 	Constant-Q chromagram
chroma_cens([y, sr, C, hop_length, fmin, …]) 	Computes the chroma variant “Chroma Energy Normalized” (CENS), following [R674badebce0d-1].
melspectrogram([y, sr, S, n_fft, …]) 	Compute a mel-scaled spectrogram.
mfcc([y, sr, S, n_mfcc, dct_type, norm]) 	Mel-frequency cepstral coefficients (MFCCs)
rmse([y, S, frame_length, hop_length, …]) 	Compute root-mean-square (RMS) energy for each frame, either from the audio samples y or from a spectrogram S.
spectral_centroid([y, sr, S, n_fft, …]) 	Compute the spectral centroid.
spectral_bandwidth([y, sr, S, n_fft, …]) 	Compute p’th-order spectral bandwidth:
spectral_contrast([y, sr, S, n_fft, …]) 	Compute spectral contrast [R6ffcc01153df-1]
spectral_flatness([y, S, n_fft, hop_length, …]) 	Compute spectral flatness
spectral_rolloff([y, sr, S, n_fft, …]) 	Compute roll-off frequency.
poly_features([y, sr, S, n_fft, hop_length, …]) 	Get coefficients of fitting an nth-order polynomial to the columns of a spectrogram.
tonnetz([y, sr, chroma]) 	Computes the tonal centroid features (tonnetz), following the method of [Recf246e5a035-1].
zero_crossing_rate(y[, frame_length, …]) 	Compute the zero-crossing rate of an audio time series.
