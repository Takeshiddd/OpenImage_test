import numpy as np
import matplotlib
from typing import Any, List
from gensim.models import word2vec, Word2Vec
from nltk.corpus import stopwords
import csv
from collections import defaultdict
from numpy.core.multiarray import ndarray

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import sys, time
import os

import pprint
import cv2
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import skimage.io as io


def class_dict():  # 引数： なし　戻り値： OpenImagesのオブジェクトクラスとIDの辞書型の値（｛ID：クラス｝）
    class_dict = {}
    with open('class-descriptions.csv', newline='') as class_csvfile:
        class_description = csv.reader(class_csvfile, delimiter=',', quotechar='"')
        for row in class_description:
            class_dict[row[0]] = row[1]
    return class_dict


def Image_dict(class_dict,
               annotations_human_bbox_PATH):  # 引数： OpenImagesのオブジェクトクラスとIDの辞書型の値（｛ID：クラス｝）　戻り値： OpenImagesのannotations-human-bboxファイルから辞書型のBBOX情報の変数を返す （こんな形→{'画像ID': [{'classname': 'クラス名', 'position': ['bboxのXmin座標','bboxのXmax座標','bboxのYmin座標','bboxのYmax座標']}, {'classname': 'クラス名', 'position': ['bboxのXmin座標','bboxのXmax座標','bboxのYmin座標','bboxのYmax座標']}...
    Image_dict = defaultdict(list)  # example: Image_dict(class_dict(), './annotations-human-bbox/test/annotations-human-bbox.csv')
    with open(annotations_human_bbox_PATH, newline='') as csvfile:
        next(csvfile)
        bbox_description = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in bbox_description:
            bbox_dict = {'classname': '', 'position': ''}
            bbox_dict['classname'] = class_dict[row[2]]
            row[4:8] = [float(a) for a in row[4:8]]
            bbox_dict['position'] = row[4:8]
            Image_dict[row[0]].append(bbox_dict)
    return Image_dict

# def Text_dict(Text_bbox_PATH, imagesdir_PATH):        # ウィンドウサイズを画像から取得する版
#     Text_dict = defaultdict(list)        # example: Image_dict(class_dict(), './annotations-human-bbox/test/annotations-human-bbox.csv')
#     with open(Text_bbox_PATH, newline='') as csvfile:
#         bbox_description = csv.reader(csvfile, delimiter=',', quotechar='"')
#         for row in bbox_description:
#             im = cv2.imread("{}/{}.jpg".format(imagesdir_PATH,row[0]))
#             windowsize = im.shape[:2]
#             bbox_dict = {}
#             bbox_dict['text'] = row[-1]
#             bbox_dict['position'] = row[1:9]
#             bbox_dict['position'][0] = float(bbox_dict['position'][0]) / windowsize[1]
#             bbox_dict['position'][2] = float(bbox_dict['position'][2]) / windowsize[1]
#             bbox_dict['position'][4] = float(bbox_dict['position'][4]) / windowsize[1]
#             bbox_dict['position'][6] = float(bbox_dict['position'][6]) / windowsize[1]
#             bbox_dict['position'][1] = float(bbox_dict['position'][1]) / windowsize[0]
#             bbox_dict['position'][3] = float(bbox_dict['position'][3]) / windowsize[0]
#             bbox_dict['position'][5] = float(bbox_dict['position'][5]) / windowsize[0]
#             bbox_dict['position'][7] = float(bbox_dict['position'][7]) / windowsize[0]
#             Text_dict[row[0]].append(bbox_dict)
#     return Text_dict

def windowsize(windowsize_PATH):
    with open(windowsize_PATH, newline='') as f:
        reader = csv.reader(f, delimiter=',', quotechar='"')
        windowsize = {}
        for row in reader:
            windowsize[row[0]] = (float(row[1]), float(row[2]))
    return windowsize        #(width, hight)

def Text_dict(Text_bbox_PATH, windowsize):
    Text_dict = defaultdict(list)        # example: Image_dict(class_dict(), './annotations-human-bbox/test/annotations-human-bbox.csv')
    with open(Text_bbox_PATH, newline='') as csvfile:
        bbox_description = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in bbox_description:
            bbox_dict = {}
            bbox_dict['text'] = row[-1]
            bbox_dict['position'] = row[1:9]
            bbox_dict['position'][0] = float(bbox_dict['position'][0]) / windowsize[row[0]][0]
            bbox_dict['position'][2] = float(bbox_dict['position'][2]) / windowsize[row[0]][0]
            bbox_dict['position'][4] = float(bbox_dict['position'][4]) / windowsize[row[0]][0]
            bbox_dict['position'][6] = float(bbox_dict['position'][6]) / windowsize[row[0]][0]
            bbox_dict['position'][1] = float(bbox_dict['position'][1]) / windowsize[row[0]][1]
            bbox_dict['position'][3] = float(bbox_dict['position'][3]) / windowsize[row[0]][1]
            bbox_dict['position'][5] = float(bbox_dict['position'][5]) / windowsize[row[0]][1]
            bbox_dict['position'][7] = float(bbox_dict['position'][7]) / windowsize[row[0]][1]
            Text_dict[row[0]].append(bbox_dict)
    return Text_dict


def inbox(Classbbox_posi,
          Textbbox_posi):  # 引数： bboxのポジション２つ（[Xmim, Xmax, Ymin, Ymax])　戻り値： ClassbboxにTextbboxが入ってたらTrue,それ以外はFalse
    if Classbbox_posi[0] < Textbbox_posi[0] < Classbbox_posi[1] and Classbbox_posi[0] < Textbbox_posi[2] < Classbbox_posi[1] and Classbbox_posi[0] < Textbbox_posi[4] < Classbbox_posi[1] and Classbbox_posi[0] < Textbbox_posi[6] < Classbbox_posi[1] \
            and Classbbox_posi[2] < Textbbox_posi[1] < Classbbox_posi[3] and Classbbox_posi[2] < Textbbox_posi[3] < Classbbox_posi[3] and Classbbox_posi[2] < Textbbox_posi[5] < Classbbox_posi[3] and Classbbox_posi[2] < Textbbox_posi[7] < Classbbox_posi[3] :
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

def position_data(model, counter):  # count関数の戻り値({("classname","text": 回数、 ...})のキーをwv化して、np.arrray型の座標にして返す
    position_data = []  # type: List[ndarray]
    GSL_freq = []
    stopWords = stopwords.words('english')
    with open("GSL_frequency.csv", 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            GSL_freq.append(row[2])
    for key in counter.keys():
        if key[0] in stopWords or key[1] in stopWords:
            print("{} or {} is not in vocabulary".format(key[0], key[1]))
        else:
            if key[0] in GSL_freq and key[1] in GSL_freq:
                try:
                    data = np.array(np.r_[model[key[0]], model[key[1]], np.array([counter[key]])])
                    position_data.append(data)
                except:
                    print("{} or {} is not in vocabulary".format(key[0], key[1]))
            else:
                print("{} or {} is not in vocabulary".format(key[0], key[1]))
    return position_data

# def position_data(model, counter):  # count関数の戻り値({("classname","text": 回数、 ...})のキーをwv化して、np.arrray型の座標にして返す   （ストップワードとGSL考慮無し版）
#     position_data = []  # type: List[ndarray]
#     for key in counter.keys():
#         try:
#             data = np.array(np.r_[model[key[0]], model[key[1]], np.array([counter[key]])])
#             position_data.append(data)
#         except:
#             print("Maybe {}, {} is not in vocabulary".format(key[0], key[1]))
#     return position_data





def wordvec_dict(model):
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





class_dict = class_dict()
Image_dict = Image_dict(class_dict, './annotations-human-bbox/train-annotations-bbox.csv')
text_dict = Text_dict('result170_1201.csv', windowsize('retult_size.csv'))
counter = count(Image_dict, text_dict)
model = word2vec.Word2Vec.load("sample2.model")
# model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)

data = position_data(model, counter)
print(data)
with open('positiondata_GSL_stopword.csv', 'w') as file:
   writer = csv.writer(file, lineterminator='\n')
   writer.writerows(data)


print(text_dict)