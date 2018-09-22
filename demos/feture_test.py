import librosa, librosa.display
import pandas as pd

original_song = "../data/test_data/Original.mp3"
cover_song = "../data/test_data/Cover.mp3"

original_signal = librosa.load(original_song)[0]
cover_signal = librosa.load(cover_song)[0]

def extract_features(signal):
    return {
        "zero_crossing_rate": librosa.feature.zero_crossing_rate(signal)[0, 0],
        "spectral_centroid": librosa.feature.spectral_centroid(signal)[0, 0],
        "spectral_bandwidth": librosa.feature.spectral_bandwidth(signal)[0, 0],
        "spectral_contrast": librosa.feature.spectral_contrast(signal)[0, 0],
        "spectral_flatness": librosa.feature.spectral_flatness(signal)[0, 0],
        "spectral_rolloff": librosa.feature.spectral_rolloff(signal)[0, 0],
        "chroma_stft": librosa.feature.chroma_stft(signal)[0, 0],
        "chroma_cqt": librosa.feature.chroma_cqt(signal)[0, 0],
        "chroma_cens": librosa.feature.chroma_cens(signal)[0, 0],
        "melspectrogram": librosa.feature.melspectrogram(signal)[0, 0],
        "mfcc": librosa.feature.mfcc(signal)[0, 0],
        "rmse": librosa.feature.rmse(signal)[0, 0],
        "poly_features": librosa.feature.poly_features(signal)[0, 0],
        "tonnetz": librosa.feature.tonnetz(signal)[0, 0]
    } 

def save_data_frame(data1, data2):
    df1 = pd.DataFrame({"original" :data1})
    df2 = pd.DataFrame({"cover" : data2})

    final_df = pd.concat([df1,df2], axis = 1)
    final_df.to_csv("../data/results/features.csv")

save_data_frame(extract_features(original_signal), extract_features(cover_signal))
