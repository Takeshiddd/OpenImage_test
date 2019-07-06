import csv
from gensim.models import word2vec
import numpy as np
from tqdm import tqdm

model = word2vec.Word2Vec.load("sample2.model")
l = []
with open('positiondata2.csv', 'r') as f:
   reader = csv.reader(f)
   for row in tqdm(reader):
       row_f = [float(n) for n in row]
       l.append([model.most_similar([np.array(row_f[0:200], dtype='f4')], [], 1)[0][0], model.most_similar([np.array(row_f[200:400], dtype='f4')], [], 1)[0][0], row_f[-1]])
with open('obj_word_num.csv', 'w') as file:
   writer = csv.writer(file, lineterminator='\n')
   writer.writerows(l)
