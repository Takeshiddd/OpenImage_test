import numpy as np
from tqdm import tqdm
from gensim.models import word2vec, Word2Vec
import matplotlib
from typing import Any, List
from gensim.models import word2vec, Word2Vec
import csv
from matplotlib import pyplot as plt
from numpy.core.multiarray import ndarray


def positions(positiondata_PATH):
    positions = []
    with open(positiondata_PATH, "r") as f:
        reader = csv.reader(f)
        for row in tqdm(reader):
            positions.append([float(data) for data in row])
    return np.array(positions)


def inpeek(x, y, box):  # 引数： bboxのポジション（[Xmim, Xmax, Ymin, Ymax])　戻り値： 入ってたらTrue,それ以外はFalse
    if box[0] < x < box[1] and box[2] < y < box[3]:
        return True
    else:
        return False


xy_positions = positions('3d_positiondata_remake.csv')
obj_word_positions = positions('positiondata_tyuhuku.csv')
model = word2vec.Word2Vec.load("sample2.model")

x1 = -10.6694
x2 = -10.4499
mat = np.c_[xy_positions, obj_word_positions]
s = []
for row in mat:
    if x1 < row[0] < x2:
        s.append(row)
a = []
for row in s:
    a.append(row[2:])



l = []
for row in tqdm(a):
    l.append([model.most_similar([np.array(row[0:200])], [], 1)[0][0],
              model.most_similar([np.array(row[200:400])], [], 1)[0][0]])

with open('./x_{}-{}_in_hist.csv'.format(x1, x2), 'w') as file:
    writer = csv.writer(file, lineterminator='\n')
    writer.writerows(l)
