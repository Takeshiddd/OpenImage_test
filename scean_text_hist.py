#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import csv
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from collections import defaultdict
from gensim.models import word2vec, Word2Vec



positions = []
model = word2vec.Word2Vec.load("sample2.model")
vocab = model.wv.vocab

with open('result170_1201.csv', newline='') as csvfile:
    bbox_description = csv.reader(csvfile, delimiter=',', quotechar='"')
    for row in bbox_description:
        if row[-1] in vocab:
            text_posi = model[row[-1]]
            positions.append(text_posi)








# new_positions = []
# for row in tqdm(positions):
#     counter = row[-1]
#     i = 0
#     while i < counter:
#         new_positions.append(np.array(row[0:400]))
#         i += 1

positions = np.array(positions)

# 主成分分析する
pca = PCA(1)



# 分析結果を元にデータセットを主成分に変換する
transformed_posi = pca.fit_transform(positions)


with open('shuseibumtext_posi.csv', 'w') as file:
   writer = csv.writer(file, lineterminator='\n')
   writer.writerows(transformed_posi)