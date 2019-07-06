import webbrowser
import csv
from tqdm import tqdm

d = {}
with open('image_ids_and_rotation(1).csv', newline='') as csvfile:
    bbox_description = csv.reader(csvfile, delimiter=',', quotechar='"')
    for row in tqdm(bbox_description):
        d[row[0]] = row[2]

IDs = []
with open('poster_the_IDs.csv', newline='') as csvfile:
    writer = csv.reader(csvfile, delimiter=',', quotechar='"')
    for row in writer:
        IDs = row

# i = 0
# for i in range[0:len(IDs)]:
#     ID_list = IDs[i:i+5]
#     for ID in ID_list:
#         webbrowser.open(d[ID])

ID_list = IDs[50:90]
# ID_list = ['0abf73c49ac29d70']
for ID in ID_list:
    webbrowser.open(d[ID])
