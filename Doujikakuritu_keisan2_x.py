# coding:utf-8
import numpy as np
import matplotlib
from tqdm import tqdm
from gensim.models import word2vec
import csv
from nltk.corpus import stopwords
matplotlib.use('TKAgg')
from sklearn.neighbors import NearestNeighbors
from operator import itemgetter

def positions(positiondata_PATH):
    positions = []
    with open(positiondata_PATH, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            positions.append([float(data) for data in row])
    return np.array(positions)

def position_count_datas(model, positions, object, text, k):
    position = np.array(np.r_[model[object], model[text]])
    a = []
    l = []
    l_a_list = []
    i = 0
    for data in positions:
        a.append(data[-1])
        l.append(np.linalg.norm(data[0:-1] - position))
    while i < k:
        index = np.argmin(np.array(l))
        l_a_list.append((l[index], a[index]))                                                                   
        del l[index]
        del a[index]
        i += 1
    return l_a_list

def Probability(l_a_list, sigma2 = 1):
    sum = 0
    for l_a in l_a_list:
        sum += l_a[1] * np.exp(-l_a[0]**2/2/sigma2)
    P = sum / (2 * np.pi) ** 200
    return P


model = word2vec.Word2Vec.load("sample2.model")
vocab = []
stopWords = stopwords.words('english')
with open("GSL_frequency.csv", 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        vocab.append(row[2])
for word in stopWords:
    if word in vocab:
        vocab.remove(word)
wv_vocab = model.wv.vocab
for word in vocab[:]:
    if not word in wv_vocab:
        vocab.remove(word)


vocab = vocab[0:-550]


GSL_words = []
GSL_positions = []
for word in tqdm(vocab):
    for word2 in vocab:
        GSL_words.append((word,word2))
        GSL_positions.append(list(np.r_[model[word],model[word2]]))

del vocab

print('loading positiondata and fitting.')
positions = positions('positiondata_reduced_1_.csv')
nbrs = NearestNeighbors(n_neighbors=5, algorithm='kd_tree').fit(positions[:,0:-1])
distances, indices = nbrs.kneighbors(GSL_positions)

del nbrs
del GSL_positions

count_data = []
for row in indices:
    list = []
    for i in row:
        list.append(positions[i][-1])
    count_data.append(list)
count_data = np.array(count_data)

sigma2 = 10

print('calculating probability')
l2 = np.power(distances, 2)
beki = - l2 / 2 / sigma2
e = np.exp(beki)

P = []
stop = len(count_data)
for i in tqdm(range(stop)):
    P.append(np.dot(e[i],count_data[i]))
P = np.array(P)
P = np.log(P / (2 * np.pi) ** 200)




word_pare_list = []
i = 0
for i in range(stop):
    word_pare_list.append([GSL_words[i][0], GSL_words[i][1], P[i]])
    i += 1
word_pare_list.sort(key=itemgetter(2))


i = 0
with open('Word_pare_P_sigma{}.csv'.format(sigma2), 'w') as file:
    writer = csv.writer(file, lineterminator='\n')
    for i in range(stop):
        writer.writerow([GSL_words[i][0], GSL_words[i][1], P[i]])
        i += 1
print('done')

#word_pare_list = []
#i = 0
#for i in range(stop):
#    word_pare_list.append([GSL_words[i][0], GSL_words[i][1], P[i]])
#    i += 1
#print('done')


#for row in word_pare_list:
#    row[2] = np.log(float(row[2]))
#word_pare_list.sort(key=itemgetter(2))


#with open('/home/kouki/Open_Images_Cord/result/P_sort/Word_pare_P_sigma{}_sorted.csv'.format(sigma2), 'w') as file:
#    writer = csv.writer(file, lineterminator='\n')
#    writer.writerows(word_pare_list)









#100枚の果果 ※間違い
# 'bus', 'apple' '1.7798433454552452e-253'     'bus', 'bus' 3.902964949033245e-240         'bottle', 'wine' 4.859922918881773e-196
# 'bottle', 'tree' 1.4839287391829614e-226    'bottle', 'track' 2.4607677137211937e-204
# 'bottle', 'bus' 9.160043424109374e-220



#'bottle', 'bus' 1.3081264033489932e-169    'bus', 'bus' 3.398941290446398e-158     'bottle', 'wine' 7.167835374449504e-159
#'bottle', 'track' 8.285451541889063e-172



#100枚の果果　
# 'bus', 'bus' 7.930863011041595e-158       'bottle', 'wine'　1.6416655218334223e-158        'taxi taxi'  9.942722114680773e-159     train train 1.3873229756924072e-159
# 'bottle', 'track' 2.3122049594964036e-160    'bus', 'apple'  2.3122049594873456e-160      'bottle', 'bus' 5.279209671345292e-171    'cat baby' 2.826591924691839e-184          'bus apple'  2.3122049594873456e-160      cake sugar 2.307803207853676e-162
