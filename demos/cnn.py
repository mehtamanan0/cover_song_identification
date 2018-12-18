import librosa, librosa.display
import numpy as np
from scipy.spatial.distance import euclidean
from sklearn.metrics.pairwise import euclidean_distances

def extract_features(signal):
    return librosa.feature.chroma_stft(signal)[:12, :180].T

def sim_matrix(original_features, cover_features):
    similarity_matrix = []
    for each in original_features:
        sm = []
        for beach in cover_features:
            sm.append(euclidean(each, beach))
        similarity_matrix.append(sm)
    return np.array(similarity_matrix)

original_song = "../data/test_data/Original.mp3"
cover_song = "../data/test_data/Cover.mp3"

original_signal = librosa.load(original_song)[0]
cover_signal = librosa.load(cover_song)[0]

original_features = extract_features(original_signal)
cover_features = extract_features(cover_signal)
mat = sim_matrix(original_features, cover_features)
