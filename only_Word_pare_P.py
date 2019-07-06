import csv




l = []
word = 'car'
with open('Word_pare_P_sigma1_sorted.csv', 'r') as f:
   reader = csv.reader(f)
   for row in reader:
       if row[0] == word:
           l.append(row)

with open('./only/{}_only_Word_pare_P_sigma1_sorted.csv'.format(word), 'w') as file:
    writer = csv.writer(file, lineterminator='\n')
    writer.writerows(l)