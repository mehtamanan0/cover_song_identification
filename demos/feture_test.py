import librosa, librosa.display

original_song = "../data/test_data/test.mp3"
cover_song = "../data/test_data/test.mp3"

def extract_features(signal):
    return [
        librosa.feature.zero_crossing_rate(signal)[0, 0],
        librosa.feature.spectral_centroid(signal)[0, 0],
    ]


