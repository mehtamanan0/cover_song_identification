import librosa, librosa.display

original_song = "../data/test_data/test.mp3"
cover_song = "../data/test_data/test.mp3"

signal = librosa.load(original_song)[0]

def extract_features(signal):
    return [
        librosa.feature.zero_crossing_rate(signal)[0, 0],
        librosa.feature.spectral_centroid(signal)[0, 0],
        librosa.feature.spectral_bandwidth(signal)[0, 0],
        librosa.feature.spectral_contrast(signal)[0, 0],
        librosa.feature.spectral_flatness(signal)[0, 0],
        librosa.feature.spectral_rolloff(signal)[0, 0],
        librosa.feature.chroma_stft(signal)[0, 0],
        librosa.feature.chroma_cqt(signal)[0, 0],
        librosa.feature.chroma_cens(signal)[0, 0],
        librosa.feature.melspectrogram(signal)[0, 0],
        librosa.feature.mfcc(signal)[0, 0],
        librosa.feature.rmse(signal)[0, 0],
        librosa.feature.poly_features(signal)[0, 0],
        librosa.feature.tonnetz(signal)[0, 0],
        librosa.feature.zero_crossing_rate(signal)[0, 0]
    ]

extract_features(signal)
