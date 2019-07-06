#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import csv
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA



def positions(positiondata_PATH):
    positions = []
    with open(positiondata_PATH, "r") as f:
        reader = csv.reader(f)
        for row in tqdm(reader):
            positions.append([float(data) for data in row])
    return np.array(positions)


positions = positions('positiondata_GSL_stopword.csv')
new_positions = []
for row in positions:
    counter = row[-1]
    i = 0
    while i < counter:
        new_positions.append(np.array(row[0:400]))
        i += 1

new_positions = np.array(new_positions)
with open('positiondata_tyuhuku.csv', 'w') as file:
   writer = csv.writer(file, lineterminator='\n')
   writer.writerows(new_positions)

object_posi = new_positions[:,0:200]
word_posi = new_positions[:,200:400]

# 主成分分析する
pca = PCA(1)



# 分析結果を元にデータセットを主成分に変換する
transformed_object_posi = pca.fit_transform(object_posi)
transformed_word_posi = pca.fit_transform(word_posi)

new_position_data = np.c_[transformed_object_posi, transformed_word_posi]

with open('3d_positiondata_remake.csv', 'w') as file:
   writer = csv.writer(file, lineterminator='\n')
   writer.writerows(new_position_data)