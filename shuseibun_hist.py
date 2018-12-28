import numpy as np
import matplotlib
from typing import Any, List
from gensim.models import word2vec, Word2Vec
import csv
from matplotlib import pyplot as plt




def positions(positiondata_PATH):
    positions = []
    with open(positiondata_PATH, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            positions.append(np.array([float(data) for data in row]))
    return positions



positions = positions('3d_positiondata_remake.csv')
model = word2vec.Word2Vec.load("sample2.model")

x = np.array(positions)[:, 0]
y = np.array(positions)[:, 1]


fig = plt.figure()
ax = fig.add_subplot(111)

H = ax.hist2d(x,y, bins=500)
ax.set_title('1st graph')
ax.set_xlabel('x')
ax.set_ylabel('y')
fig.colorbar(H[3],ax=ax)
plt.show()
