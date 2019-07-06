import csv
from operator import itemgetter
import numpy as np

sigma = 3
word_pare_list = []
with open('Word_pare_P_sigma{}.csv'.format(sigma), "r") as f:
    reader = csv.reader(f)
    for row in reader:
        word_pare_list.append(row)

for row in word_pare_list:
    row[2] = float(row[2])
word_pare_list.sort(key=itemgetter(2))


with open('Word_pare_P_sigma{}_sorted.csv'.format(sigma), 'w') as file:
    writer = csv.writer(file, lineterminator='\n')
    writer.writerows(word_pare_list)
