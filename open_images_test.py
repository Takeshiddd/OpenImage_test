import numpy as np
import matplotlib
from typing import Any
from gensim.models import word2vec, Word2Vec
from gensim.models import KeyedVectors
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import sys, time
import os
import csv
from collections import defaultdict
import pprint
import time

matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import skimage.io as io


def class_dict():  # 引数： なし　戻り値： OpenImagesのオブジェクトクラスとIDの辞書型の値（｛ID：クラス｝）
    class_dict = {}
    with open('class-descriptions-boxable.csv', newline='') as class_csvfile:
        class_description = csv.reader(class_csvfile, delimiter=',', quotechar='"')
        for row in class_description:
            class_dict[row[0]] = row[1]
    return class_dict


def Image_dict(class_dict,
               annotations_human_bbox_PATH):  # 引数： OpenImagesのオブジェクトクラスとIDの辞書型の値（｛ID：クラス｝）　戻り値： OpenImagesのannotations-human-bboxファイルから辞書型のBBOX情報の変数を返す （こんな形→{'画像ID': [{'classname': 'クラス名', 'position': ['bboxのXmin座標','bboxのXmax座標','bboxのYmin座標','bboxのYmax座標']}, {'classname': 'クラス名', 'position': ['bboxのXmin座標','bboxのXmax座標','bboxのYmin座標','bboxのYmax座標']}...
    Image_dict = defaultdict(
        list)  # example: Image_dict(class_dict(), './annotations-human-bbox/test/annotations-human-bbox.csv')
    with open(annotations_human_bbox_PATH, newline='') as csvfile:
        next(csvfile)
        bbox_description = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in bbox_description:
            bbox_dict = {'classname': '', 'position': ''}
            bbox_dict['classname'] = class_dict[row[2]]
            bbox_dict['position'] = row[4:8]
            Image_dict[row[0]].append(bbox_dict)
    return Image_dict


def inbox(Classbbox, Textbbox):
    if Classbbox[0] < Textbbox[0] < Classbbox[1] and Classbbox[0] < Textbbox[1] < Classbbox[1] and Classbbox[2] < \
            Textbbox[2] < Classbbox[3] and Classbbox[2] < Textbbox[3] < Classbbox[3]:
        return True
    else:
        return False


def count(Image_dict,
          Text_dict):  # ImagedictとTextdictから、classbboxにtextbboxが入ったペアと入った回数を({("classname","text": 回数、 ...})   辞書型で返す　※This function requaiers function "inbox". inbox関数も一緒に定義しましょう。
    counter = defaultdict(int)
    for ImageID in Image_dict.keys():
        for classbbox in Image_dict[ImageID]:
            for textbbox in Text_dict[ImageID]:
                if inbox(classbbox['position'], textbbox['position']):
                    counter[(classbbox['classname'].lower(), textbbox['text'].lower())] += 1
    return counter


def position_data(model, counter):  # count関数の戻り値({("classname","text": 回数、 ...})のキーをwv化して、np.arrray型の座標のリスト型にして返す
    position_data = []  # type: List[ndarray]
    for key in counter.keys():
        try:
            data = np.array(np.r_[model[key[0]], model[key[1]], counter[key]])
            position_data.append(data)
        except:
            print("Maybe {}, {} is not in vocabulary".format(key[0], key[1]))
    return position_data

def wordvec_dict(model):    # 引数： word2vecモデルファイル　戻り値： 単語とベクトルの辞書{”単語”： [ベクトル]}
    vocab = model.wv.vocab
    wordvec_dict = {}
    for word in vocab.keys():
        wordvec_dict[word] = model[word]
    return wordvec_dict



def extend_wordvecs(wordvec_dict):   # 引数： 点後と単語ベクトルの辞書{”単語”： [ベクトル]}　戻り値： 各軸の最大値と最小値をnparray型の配列で返す  ※extnd_wordvecs(wordvec_dict)[0] = min, extnd_wordvecs(wordvec_dict)[1] = max
    position_list = list(wordvec_dict.values())
    position_array = np.array(position_list)
    print(position_array.shape)
    return np.array([np.min(position_array, axis=0), np.max(position_array, axis=0)])




def serch_neighbourhood(positiondata):
    list = []
    list2 = []
    for posi in positiondata:
        lis = []
        lis2 = []
        for posi2 in positiondata:
            de = np.linalg.norm(posi - posi2)
            if de != 0:
                lis.append(de)
                lis2.append([posi, posi2])
        list.append(min(lis))
        list2.append(lis2[lis.index(min(lis))])
    return (min(list), list2[list.index(min(list))])



def Text_dict(Text_bbox_PATH, imagesdir_PATH):
    Text_dict = defaultdict(list)        # example: Image_dict(class_dict(), './annotations-human-bbox/test/annotations-human-bbox.csv')
    with open(Text_bbox_PATH, newline='') as csvfile:
        bbox_description = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in bbox_description:
            im = cv2.imread("{}/{}.jpg".format(imagesdir_PATH,row[0]))
            windowsize = im.shape[:2]
            bbox_dict = {}
            bbox_dict['Text'] = row[-1]
            bbox_dict['position'] = row[1:9]
            bbox_dict['position'][0] = float(bbox_dict['position'][0]) / windowsize[1]
            bbox_dict['position'][2] = float(bbox_dict['position'][2]) / windowsize[1]
            bbox_dict['position'][4] = float(bbox_dict['position'][4]) / windowsize[1]
            bbox_dict['position'][6] = float(bbox_dict['position'][6]) / windowsize[1]
            bbox_dict['position'][1] = float(bbox_dict['position'][1]) / windowsize[0]
            bbox_dict['position'][3] = float(bbox_dict['position'][3]) / windowsize[0]
            bbox_dict['position'][5] = float(bbox_dict['position'][5]) / windowsize[0]
            bbox_dict['position'][7] = float(bbox_dict['position'][7]) / windowsize[0]
            Text_dict[row[0]].append(bbox_dict)
    return Text_dict


# model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)
model = word2vec.Word2Vec.load("sample.model")



import cv2

Imagedict = {'a': [{'classname': 'apple', 'position': [0,1,0,1]},{'classname': 'pen', 'position':[0,1,0,1]}],
             'b': [{'classname':'bus', 'position': [0,1,0,1]},{'classname':'train', 'position':[0,1,0,1]}]}
Textdict ={'a': [{'text':'hello', 'position': [0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]}],
             'b': [{'text':'bye', 'position': [0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]}]}

c = count(Imagedict, Textdict)
print(c)
p = position_data(model, c)

print(p)


with open('positiondata.csv', 'w') as file:
   writer = csv.writer(file, lineterminator='\n')
   writer.writerows(p)

# counter = {('apple', 'hello'): 1, ('pen', 'apple'): 1, ('bus', 'bye'): 1, ('train', 'bye'): 1}
# p = position_data(model, counter)
#
# print(np.shape(p))






# Image_dict = Image_dict(class_dict(), './annotations-human-bbox/train-annotations-bbox.csv')
# print('0a0ae80d5fecd187' in Image_dict.keys())
# import cv2
# im = cv2.imread("./result_check/100_images/0a0ae80d5fecd187.jpg")
# print(im.shape)
# pprint.pprint(Image_dict['0a0ae80d5fecd187'])
# cv2.namedWindow('window')
#
#
# cv2.imshow('window', im)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()




# import  cv2
# Text_bbox_PATH = "./result_check/100_texts/000a1cb9f7f9f11b.txt"
# imagesdir_PATH = "./result_check/100_images"
# pprint.pprint(Text_dict(Text_bbox_PATH, imagesdir_PATH))



# import cv2
#
# im = cv2.imread("./result_check/100_images/0c7c0a9fbcdff135.jpg")
# print(im.shape)
# cv2.namedWindow('window')
#
#
# cv2.imshow('window', im)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()





# t1 = time.time()
# a = np.exp(-1/2*np.linalg.norm(model["train"]))
# print(a)
#
# t2 = time.time()
#
# print(t2-t1)



#
# wordvecdict = wordvec_dict(model)
#
#
# pd = np.array([[0,1,2,3],[0,1,2,4],[0,1,2,3.1],[0,1,2,3]])
# a = serch_neighbourhood(pd)
# # a = serch_neighbourhood(wordvecdict.values())
#
# print(a[1])
# print(a[0])










# wordvec_dict = wordvec_dict(model)
#
# print(min(extnd_wordvecs(wordvec_dict)[0]))
# print(max(extnd_wordvecs(wordvec_dict)[1]))










# print(model.wv["train"])

# pprint.pprint(Image_dict(class_dict(), './annotations-human-bbox/test/annotations-human-bbox.csv'))
