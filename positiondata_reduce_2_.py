import csv

i = 0
list = []
with open('GSL_frequency.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        i += 1
        list.append(row)
        if i > 100:
            break
        
print(list[i-1])
with open('GSL_frequency_quite_reduced_for_test.csv', 'w') as file:
    writer = csv.writer(file, lineterminator='\n')
    writer.writerows(list)
