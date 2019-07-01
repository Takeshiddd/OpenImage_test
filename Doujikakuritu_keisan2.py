import numpy as np
import matplotlib
from tqdm import tqdm
from typing import Any, List
from gensim.models import word2vec, Word2Vec
from gensim.models import KeyedVectors
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import sys, time
import os
import csv
from nltk.corpus import stopwords
from collections import defaultdict
import pprint
import cv2
from numpy.core.multiarray import ndarray

matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import skimage.io as io


def positions(positiondata_PATH):
    positions = []
    with open(positiondata_PATH, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            positions.append(np.array([float(data) for data in row]))
    return positions

def position_count_datas(model, positions, object, text, k):
    vocab = model.wv.vocab
    if not object in vocab:
        print('{} is not in vocabulary.'.format(object))
        return -1
    elif not text in vocab:
        print('{} is not in vocabulary.'.format(text))
        return -1
    else:
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


vocab = []
stopWords = stopwords.words('english')
with open("GSL_frequency.csv", 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        vocab.append(row[2])
for word in stopWords:
    if word in vocab:
        vocab.remove(word)



positions = positions('positiondata_GSL_stopword.csv')
model = word2vec.Word2Vec.load("sample2.model")
p_dict = {}
for object in tqdm(vocab):
    for text in vocab:
        pds = position_count_datas(model, positions, object, text, k=5)
        if pds != -1:
            P = Probability(pds)
            p_dict[(object,text)] = P


with open("p_dict.csv", 'w') as f:
    writer = csv.writer(f, lineterminator='\n')
    list = []
    for key in p_dict.keys():
        list.append([key[0], key[1], str(p_dict[key])])
    writer.writerows(list)

i = 0
while i < 50:
    max_k = max(p_dict, key=p_dict.get)
    print(max_k, p_dict[max_k])
    i += 1
    del p_dict[max_k]
#100枚の果果 ※間違い
# 'bus', 'apple' '1.7798433454552452e-253'     'bus', 'bus' 3.902964949033245e-240         'bottle', 'wine' 4.859922918881773e-196
# 'bottle', 'tree' 1.4839287391829614e-226    'bottle', 'track' 2.4607677137211937e-204
# 'bottle', 'bus' 9.160043424109374e-220



#'bottle', 'bus' 1.3081264033489932e-169    'bus', 'bus' 3.398941290446398e-158     'bottle', 'wine' 7.167835374449504e-159
#'bottle', 'track' 8.285451541889063e-172



#100枚の果果　
# 'bus', 'bus' 7.930863011041595e-158       'bottle', 'wine'　1.6416655218334223e-158        'taxi taxi'  9.942722114680773e-159     train train 1.3873229756924072e-159
# 'bottle', 'track' 2.3122049594964036e-160    'bus', 'apple'  2.3122049594873456e-160      'bottle', 'bus' 5.279209671345292e-171    'cat baby' 2.826591924691839e-184          'bus apple'  2.3122049594873456e-160      cake sugar 2.307803207853676e-162
