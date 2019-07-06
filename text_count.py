
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

model = word2vec.Word2Vec.load("sample2.model")
vocab = []
stopWords = stopwords.words('english')
with open("GSL_frequency.csv", 'r') as f:
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
Stopwords = stopwords.words('english')
text = []

with open('result170_1201.csv', newline='') as csvfile:
    bbox_description = csv.reader(csvfile, delimiter=',', quotechar='"')
    for row in bbox_description:
        if row[-1] in vocab: # and row[-1] not in Stopwords and len(row[-1]) != 1:
                text.append(row[-1])
text_dict = collections.Counter(text)
l = []


for k, v in sorted(text_dict.items(), key=lambda x: -x[1]):
    l.append([str(k), str(v)])

with open('text_count_GSL_SW.csv', 'w') as file:
   writer = csv.writer(file, lineterminator='\n')
   writer.writerows(l)