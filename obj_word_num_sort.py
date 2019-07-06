import csv
from operator import itemgetter
import numpy as np
from nltk.corpus import stopwords


vocab = []
stopWords = stopwords.words('english')
with open("GSL_frequency.csv", 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        vocab.append(row[2])
for word in stopWords:
    if word in vocab:
        vocab.remove(word)

word_pare_list = []
with open('obj_word_num.csv', "r") as f:
    reader = csv.reader(f)
    for row in reader:
        if row[0] in vocab and row[1] in vocab:
            word_pare_list.append(row)

for row in word_pare_list:
    row[2] = float(row[2])
word_pare_list.sort(key=itemgetter(2))


with open('obj_word_num_sorted.csv', 'w') as file:
    writer = csv.writer(file, lineterminator='\n')
    writer.writerows(word_pare_list)
