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


mat = np.c_[xy_positions, obj_word_positions]
inpeek_obj_word_list = []
for row in mat:
    if inpeek(row[0], row[1], box = [2.09638, 2.1411, -4.44273, -4.39254]) == True:
        obj_vec = row[2:202]
        word_vec = row[202:402]
        inpeek_obj_word_list.append((model.most_similar([obj_vec], [], 1)[0][0], model.most_similar([word_vec], [], 1)[0][0]))

print(inpeek_obj_word_list)


