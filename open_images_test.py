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
    with open('class-descriptions.csv', newline='') as class_csvfile:
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
    counter = defaultdict(set)
    for ImageID in Image_dict.keys():
        for classbbox in Image_dict[ImageID]:
            for textbbox in Text_dict[ImageID]:
                if inbox(classbbox['position'], textbbox['position']):
                    counter[(classbbox['classname'], textbbox['text'])] += 1
    return counter


def position_data(counter):  # count関数の戻り値({("classname","text": 回数、 ...})のキーをwv化して、np.arrray型の座標にして返す
    position_data = []  # type: List[ndarray]
    for key in counter.keys():
        try:
            data = np.array(np.r_[model[key[0]], model[key[1]], counter[key]])
            position_data.append(data)
            return position_data
        except:
            print("Maybe {}, {} is not in vocabulary".format(key[0], key[1]))


def wordvec_dict(model):    # 引数： word2vecモデルファイル　戻り値： 単語とベクトルの辞書{”単語”： [ベクトル]}
    vocab = model.wv.vocab
    wordvec_dict = {}
    for word in vocab.keys():
        wordvec_dict[word] = model[word]
    return wordvec_dict










# model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)
model = word2vec.Word2Vec.load("sample.model")



def extnd_wordvecs(wordvec_dict):   # 引数： 点後と単語ベクトルの辞書{”単語”： [ベクトル]}　戻り値： 各軸の最大値と最小値のリスト   未完成
    position_list = list(wordvec_dict.values())
    position_array = np.array(position_list)
    print(position_array.shape)
    print(np.max(position_array, axis=0))
    print(np.min(position_array, axis=0))








# print(model.wv["train"])

# pprint.pprint(Image_dict(class_dict(), './annotations-human-bbox/test/annotations-human-bbox.csv'))
