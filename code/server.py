from pydub import AudioSegment
from pydub.playback import play
from keras.models import load_model
import operator
import io
import keras
import os
import librosa, librosa.display
import numpy as np
from scipy.spatial.distance import euclidean
from sklearn.metrics.pairwise import euclidean_distances
import sys
from multiprocessing import Pool
from flask import Flask, request, jsonify #import main Flask class and request object

def oti_func(original_features, cover_features):
    profile1 = np.sum(original_features.T, axis = 1)
    profile2 = np.sum(cover_features.T, axis = 1)
    oti = [0] * 12
    for i in range(profile2.shape[0]):
        oti[i] = np.dot(profile1, np.roll(profile2, i))
    oti.sort(reverse=True)
    newmusic = np.roll(cover_features.T, int(oti[0]))
    return newmusic.T

def load_extract(path):
    signal = librosa.load(path)[0]
    return librosa.feature.chroma_stft(signal)[:12, :180].T

def sim_matrix(original_features, cover_features):
    similarity_matrix = []
    for each in original_features:
        sm = []
        for beach in cover_features:
            sm.append(euclidean(each, beach))
        similarity_matrix.append(sm)
    return np.array(similarity_matrix)

directory = "../data/database"
db_lis = os.listdir("../data/database")

db_dic = {}

for each in db_lis:
    db_dic[each] = load_extract(os.path.join(directory, each))

model = load_model("../data/models/model_1637.h5")
model._make_predict_function()

app = Flask(__name__) #create the Flask app

f_dic = {}

@app.route('/classify', methods = ['POST'])
def classify():
    #data = request.headers['fileNam']
    data = request.stream.read()
    with open('cover.wav', mode='bx') as f:
        f.write(data)
    cover_features = load_extract("cover.wav")
    for k, v in db_dic.items():
        original_features = v
        oti_cover_features = oti_func(original_features, cover_features)
        mat = sim_matrix(original_features, oti_cover_features)
        if mat.shape[0] < 180:
            mat = np.pad(mat, ((0,180 - mat.shape[0]),(0,0)), mode = 'constant', constant_values=0)
        if mat.shape[1] < 180:
            mat = np.pad(mat, ((0,0),(0,180 - mat.shape[1])), mode = 'constant', constant_values=0)
        mat = mat.reshape(1,180,180,1)
        f_dic[k] = model.predict(mat)[0][1]
    os.remove("cover.wav")
    l = sorted(f_dic.items(), key=operator.itemgetter(1), reverse=True)
    print(l)
    return jsonify({"result" : l[0][0]})

if __name__ == "__main__":
    app.run(host= '0.0.0.0')
