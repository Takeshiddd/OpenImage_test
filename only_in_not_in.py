import csv
from operator import itemgetter

l = []
l1 = []
l2 =[]
word1 = 'man'
word2 = 'woman'
with open('./only/{}_only_Word_pare_P_sigma1_sorted.csv'.format(word1), 'r') as f1:
   reader = csv.reader(f1)
   for row in reader:
       l1.append(row)
with open('./only/{}_only_Word_pare_P_sigma1_sorted.csv'.format(word2), 'r') as f2:
   reader = csv.reader(f2)
   for row in reader:
        l2.append(row)


l2_word_list = []
for row2 in l2:
    l2_word_list.append(row2[1])
for row in l1:
    if row[1] not in l2_word_list:
        l.append([row[1],float(row[2])])
l.sort(key=itemgetter(1))



with open('./only_in_not_in/only_in_{}_not_in_{}.csv'.format(word1, word2), 'w') as file:
    writer = csv.writer(file, lineterminator='\n')
    writer.writerows(l)
