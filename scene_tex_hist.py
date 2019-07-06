import numpy as np
import matplotlib
from typing import Any, List
from gensim.models import word2vec, Word2Vec
import csv
from matplotlib import pyplot as plt


text_list = []
model = word2vec.Word2Vec.load("sample2.model")
vocab = model.wv.vocab

with open('text_count.csv', newline='') as csvfile:
    bbox_description = csv.reader(csvfile, delimiter=',', quotechar='"')
    for row in bbox_description:
        if row[0] in vocab:
            text_list.append(int(row[-1]))
print(len(text_list))
print(text_list)

fig = plt.figure()
ax = fig.add_subplot(111)

# H = ax.hist2d(x,y, bins=100)
H = ax.hist(text_list, bins=100, normed=True, color='red', alpha=0.5)
ax.set_title('1st graph')
ax.set_xlabel('x')
# plt.xticks(color="None")
ax.set_ylabel('y')
# fig.colorbar(H[3],ax=ax)
plt.show()