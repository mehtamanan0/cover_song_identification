import tensorflow as tf
import sys
import os
from PIL import Image
# Disable tensorflow compilation warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf                                               
import librosa, librosa.display                         
import numpy as np                                      
from scipy.spatial.distance import euclidean            
from sklearn.metrics.pairwise import euclidean_distances                                
import io

original_song = sys.argv[1]
cover_song = sys.argv[2]

def oti_func(original_features, cover_features):
    profile1 = np.sum(original_features.T, axis = 1)
    profile2 = np.sum(cover_features.T, axis = 1)
    oti = [0] * 12
    for i in range(profile2.shape[0]):
        oti[i] = np.dot(profile1, np.roll(profile2, i))
    oti.sort(reverse=True)
    newmusic = np.roll(cover_features.T, int(oti[0]))
    return newmusic.T

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

original_signal = librosa.load(original_song)[0]
cover_signal = librosa.load(cover_song)[0]

original_features = extract_features(original_signal)
cover_features = extract_features(cover_signal)

oti_cover_features = oti_func(original_features, cover_features)

mat = sim_matrix(original_features, oti_cover_features)

if mat.shape[0] < 180:
    mat = np.pad(mat, ((0,180 - mat.shape[0]),(0,0)), mode = 'constant', constant_values=0)
if mat.shape[1] < 180:
    mat = np.pad(mat, ((0,0),(0,180 - mat.shape[1])), mode = 'constant', constant_values=0)

img = Image.fromarray(np.uint8(mat * 255) , 'L')
imgByteArr = io.BytesIO()
img.save(imgByteArr, format='JPEG')

imgByteArr = imgByteArr.getvalue()


if imgByteArr:
    # Read the image_data
    image_data = imgByteArr
    #image_data = cv2.imencode('.jpg', image)[1].tostring()
    # Loads label file, strips off carriage return
    label_lines = [line.rstrip() for line
                       in tf.gfile.GFile("tf_files/retrained_labels.txt")]

    # Unpersists graph from file
    with tf.gfile.FastGFile("tf_files/retrained_graph.pb", 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

    with tf.Session() as sess:
        # Feed the image_data as input to the graph and get first prediction
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        predictions = sess.run(softmax_tensor, \
                 {'DecodeJpeg/contents:0': image_data})
        # Sort to show labels of first prediction in order of confidence
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
        for node_id in top_k:
            human_string = label_lines[node_id]
            score = predictions[0][node_id]
            print('%s (score = %.5f)' % (human_string, score))
