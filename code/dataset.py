import librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import euclidean
from sklearn.metrics.pairwise import euclidean_distances
from numpy import genfromtxt
import os

def oti_func(original_features, cover_features):
    profile1 = np.sum(original_features.T, axis = 1)
    profile2 = np.sum(cover_features.T, axis = 1)
    oti = [0] * 12
    for i in range(profile2.shape[0]):
        oti[i] = np.dot(profile1, np.roll(profile2, i))
    oti.sort(reverse=True)
    newmusic = np.roll(cover_features.T, int(oti[0]))
    return newmusic.T

def sim_matrix(original_features, cover_features):
    similarity_matrix = []
    for each in original_features:
        sm = []
        for beach in cover_features:
            sm.append(euclidean(each, beach))
        similarity_matrix.append(sm)
    return np.array(similarity_matrix)

directory = "../data/mirex/"

def pair(directory):
    for filename in sorted(os.listdir(directory)):
        if filename.endswith(".csv"):
            cov = os.path.join(directory, filename)
            if cov.split("_")[2].split(".")[0] == '01':
                oname = cov
                original_features = genfromtxt(cov, delimiter=',').T
            else:
                cover_features = genfromtxt(cov, delimiter=',').T
                oti_cover_features = oti_func(original_features, cover_features)
                mat = sim_matrix(original_features, oti_cover_features)
                np.savetxt("../data/csm/pair/{}{}.csv".format(oname.split("_")[1], filename.split("_")[2].split(".")[0]), mat, delimiter=",")
        else:
            continue

def npair(directory):
    c = 0
    for filename in sorted(os.listdir(directory)):
        if c == 10000:
            break
        if filename.endswith(".csv"):
            cov = os.path.join(directory, filename)
            if cov.split("_")[2].split(".")[0] == '01':
                oname = cov
                original_features = genfromtxt(cov, delimiter=',').T
                dirlis = [x for x in os.listdir(directory) if x.endswith(".csv") and x.split("_")[2].split(".")[0] != '01' and x.split("_")[1] != oname.split("_")[1]]
                for each in dirlis:
                    path = os.path.join(directory, each)
                    cover_features = genfromtxt(path, delimiter=',').T
                    oti_cover_features = oti_func(original_features, cover_features)
                    mat = sim_matrix(original_features, oti_cover_features)
                    print("{}_{}_{}.csv".format(oname.split("_")[1], each.split("_")[1], each.split("_")[2].split(".")[0]))
                    np.savetxt("../data/csm/npair/{}_{}_{}.csv".format(oname.split("_")[1], each.split("_")[1], each.split("_")[2].split(".")[0]), mat, delimiter=",")
                    c += 1


npair(directory)
