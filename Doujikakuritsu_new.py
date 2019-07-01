# coding:utf-8
import numpy as np
import matplotlib
from tqdm import tqdm
from gensim.models import word2vec
import csv
from nltk.corpus import stopwords
matplotlib.use('TKAgg')
from sklearn.neighbors import NearestNeighbors
from operator import itemgetter

def positions(positiondata_PATH):
    positions = []
    with open(positiondata_PATH, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            positions.append([float(data) for data in row])
    return np.array(positions)

def position_count_datas(model, positions, object, text, k):
    position = np.array(np.r_[model[object], model[text]])
    a = []
    l = []
    l_a_list = []
    i = 0
    for data in positions:
        a.append(data[-1])
        l.append(np.linalg.norm(data[0:-1] - position))
    while i < k:
        index = np.argmin(np.array(l))
        l_a_list.append((l[index], a[index]))
        del l[index]
        del a[index]
        i += 1
    return l_a_list

def Probability(l_a_list, sigma2 = 1):
    sum = 0
    for l_a in l_a_list:
        sum += l_a[1] * np.exp(-l_a[0]**2/2/sigma2)
    P = sum / (2 * np.pi) ** 200
    return P

def GSL_words_positions(model, GSL_PATH):
    vocab = []
    stopWords = stopwords.words('english')
    with open(GSL_PATH, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            vocab.append(row[2])
    for word in stopWords:
        if word in vocab:
            vocab.remove(word)
    wv_vocab = model.wv.vocab
    for word in vocab[:]:
        if not word in wv_vocab:
            vocab.remove(word)

    vocab = vocab[0:-550]

    GSL_words = []
    GSL_positions = []
    for word in tqdm(vocab):
        for word2 in vocab:
            GSL_words.append((word,word2))
            GSL_positions.append(list(np.r_[model[word],model[word2]]))
    return GSL_words, GSL_positions

def k_nearest_neighbor(positions, GSL_positions, k):
    print('loading positiondata and fitting.')
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='kd_tree').fit(positions[:,0:-1])
    distances, indices = nbrs.kneighbors(GSL_positions)
    return distances, indices

def count(positions, indices):
    count_data = []
    for row in indices:
        list = []
        for i in row:
            list.append(positions[i][-1])
        count_data.append(list)
    count_data = np.array(count_data)
    return count_data

def calculate_probability(distances, count_data, sigma2, dim):
    print('calculating probability')
    l2 = np.power(distances, 2)
    beki = - l2 / 2 / sigma2
    e = np.exp(beki)
    P = []
    stop = len(count_data)
    for i in tqdm(range(stop)):
        P.append(np.dot(e[i],count_data[i]))
    P = np.array(P / (2 *sigma2 *np.pi) ** (dim / 2))
    return P

    def calculate_joint_probability(distances, sigma2, dim):
        print('calculating probability')
    l2 = np.power(distances, 2)
    beki = - l2 / 2 / sigma2
    e = np.exp(beki)
    P = []
    stop = len(distances)
    for i in tqdm(range(stop)):
        P = np.prod(e, axis=1) / (2 *sigma2 *np.pi) ** (dim / 4)
    return P

def output_result(result_PATH, GSL_words, P):
    word_pare_list = []
    stop = len(GSL_words)
    for i in range(stop):
        word_pare_list.append([GSL_words[i][0], GSL_words[i][1], P[i]])
    word_pare_list.sort(key=itemgetter(2))
    with open(result_PATH, 'w') as file:
        writer = csv.writer(file, lineterminator='\n')
        for i in range(stop):
            writer.writerow([GSL_words[i][0], GSL_words[i][1], P[i]])
    print('done')

# main #
model = word2vec.Word2Vec.load("sample2.model")
GSL_words, GSL_positions = GSL_words_positions(model, "GSL_frequency.csv")
positions = positions('positiondata_reduced_1_.csv')
distances, indices = k_nearest_neighbor(positions, GSL_positions, 5)
count_data = count(positions, indices)
sigma2 = 1
dim = 400
P = calculate_probability(distances, count_data, sigma2, dim)

# 条件付き確率の分母を計算 #
distances, indices = k_nearest_neighbor(positions, list(set(GSL_positions[:,0:200])), 5)
P_joint = calculate_joint_probability(distances, sigma2, dim)

# 条件付き確率を計算 #````````````````````````````````````````````````````
j = 0
for i in range(len(P)):
    P[i] = P[i] / P_joint[j]
    if j%1545 == 0:
        j+=1

# result_PATH = 'Word_pare_P_sigma{}.csv'.format(sigma2)
result_PATH = 'test_word_pare_joint.csv'
output_result(result_PATH, GSL_words, P)