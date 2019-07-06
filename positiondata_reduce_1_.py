import csv

i = 0
j = 0
list = []
with open("positiondata2.csv", 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        j += 1
        if float(row[-1]) != 1:
            list.append(row)
            i += 1
print(i)
print(j)
with open('positiondata_reduced_1_.csv', 'w') as file:
    writer = csv.writer(file, lineterminator='\n')
    writer.writerows(list)
