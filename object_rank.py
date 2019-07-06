

from nltk.corpus import stopwords
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import csv
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from collections import defaultdict
from gensim.models import word2vec, Word2Vec
import collections


def class_dict():  # 引数： なし　戻り値： OpenImagesのオブジェクトクラスとIDの辞書型の値（｛ID：クラス｝）
    class_dict = {}
    with open('class-descriptions.csv', newline='') as class_csvfile:
        class_description = csv.reader(class_csvfile, delimiter=',', quotechar='"')
        for row in class_description:
            class_dict[row[0]] = row[1]
    return class_dict

dataset = 'train'
cd = class_dict()
Stopwords = stopwords.words('english')
model = word2vec.Word2Vec.load("sample2.model")
vocab = model.wv.vocab
text = []
with open('./annotations-human-bbox/{}-annotations-bbox.csv'.format(dataset), newline='') as csvfile:
    next(csvfile)
    bbox_description = csv.reader(csvfile, delimiter=',', quotechar='"')
    for row in bbox_description:
        if str.lower(cd[row[2]]) in vocab:  # and row[-1] not in Stopwords and len(row[-1]) != 1:
            text.append(str.lower(cd[row[2]]))
text_dict = collections.Counter(text)

with open('object_rank_{}.csv'.format(dataset), 'w') as file:
    writer = csv.writer(file, lineterminator='\n')
    for k, v in sorted(text_dict.items(), key=lambda x: -x[1]):
        writer.writerow([str(k), str(v)])
