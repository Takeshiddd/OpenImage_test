import csv
from gensim.models import word2vec
import numpy as np


model = word2vec.Word2Vec.load("sample2.model")
l = []
with open('positiondata_reduced_1_.csv', 'r') as f:
   reader = csv.reader(f)
   for row in reader:
       if 300 > float(row[-1]) > 200:
           row_f = [float(n) for n in row]
           l.append([model.most_similar([np.array(row_f[0:200], dtype='f4')], [], 1)[0][0], model.most_similar([np.array(row_f[200:400], dtype='f4')], [], 1)[0][0], row_f[-1]])

print(l)