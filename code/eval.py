import operator
import pprint
import os
import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean
from sklearn.metrics.pairwise import euclidean_distances
import sys
from multiprocessing import Pool
from keras.models import load_model
from tqdm import tqdm
pp = pprint.PrettyPrinter(indent=4)
model = load_model('../data/models/model_1637.h5')

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
    return pd.read_csv(path, header=None).values.T

def sim_matrix(original_features, cover_features):
    similarity_matrix = []
    for each in original_features:
        sm = []
        for beach in cover_features:
            sm.append(euclidean(each, beach))
        similarity_matrix.append(sm)
    return np.array(similarity_matrix)

def eval_metrics():
    original_files = []
    pair_file = []
    directory = '../data/mirex-test/'
    t_cover = 0
    mri = 0
    mr1 = 0
    mnit10 = 0
    apc = 0
    for each in sorted(os.listdir(directory))[:22]:
        if each.split("_")[-1].split(".")[0] == '01':
            original_files.append(os.path.join(directory, each))
            pair_file.append(os.path.join(directory, each))
        else:
            pair_file.append(os.path.join(directory, each))

    for cov in original_files:
        p_bar = tqdm(total = len(pair_file))
        cover_features = load_extract(cov)
        dic = {}
        for ncov in pair_file:
            if cov == ncov:
                continue
            p_bar.update(1)
            original_features = load_extract(ncov)
            oti_cov = oti_func(original_features, cover_features)
            mat = sim_matrix(original_features, oti_cov)[:180, :180]
            if mat.shape[0] < 180:
                mat = np.pad(mat, ((0,180 - mat.shape[0]),(0,0)), mode = 'constant', constant_values=0)
            if mat.shape[1] < 180:
                mat = np.pad(mat, ((0,0),(0,180 - mat.shape[1])), mode = 'constant', constant_values=0)
            dic[ncov.split("/")[-1].split(".")[0]] = model.predict(mat.reshape(1,180,180,1))[0][1]
        l = sorted(dic.items(), key=operator.itemgetter(1), reverse=True)[:10]
        print(cov.split("/")[-1].split("_")[1])
        pp.pprint(l)
        flag = 1
        rank = 0
        ap = 0
        for i, c in enumerate(l):
            if cov.split("/")[-1].split("_")[1] == c[0].split("_")[1]:
                if flag:
                    flag = 0
                    rank = i + 1
                    print("rank : {}".format(rank))
                t_cover += 1
                ap += (c[1] * 1)
        apc += ap/11
        print("average precision : {}".format(apc))
        print("cover count: {}".format(t_cover))
        mri += 1/rank
        p_bar.close()
    mean_ap = apc/len(original_files)
    mr1 = mri/len(original_files)
    mnit10 = t_cover/3300
    return mnit10, mr1, mean_ap

a, b, c = eval_metrics()
print("mnit10 : {} mr1 : {} map : {}".format (a,b,c))
