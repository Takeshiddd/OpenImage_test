import csv
from operator import itemgetter

l = []
l1 = []
l2 =[]
word1 = 'boat'
word2 = 'car'
with open('./only/{}_only_Word_pare_P_sigma1_sorted.csv'.format(word1), 'r') as f1:
   reader = csv.reader(f1)
   for row in reader:
       l1.append(row)
with open('./only/{}_only_Word_pare_P_sigma1_sorted.csv'.format(word2), 'r') as f2:
   reader = csv.reader(f2)
   for row in reader:
        l2.append(row)


for row in l1:
    for row2 in l2:
        if row[1] == row2[1]:
            l.append([row[1],row2[1], float(row[2]) - float(row2[2])])
l.sort(key=itemgetter(2))



with open('./hikaku/{}-{}_beki_sabun.csv'.format(word1, word2), 'w') as file:
    writer = csv.writer(file, lineterminator='\n')
    writer.writerows(l)