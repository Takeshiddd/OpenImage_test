import numpy as np
import matplotlib
from typing import Any, List
from gensim.models import word2vec, Word2Vec
from nltk.corpus import stopwords
from gensim.models import KeyedVectors
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import sys, time
import os
import csv
from collections import defaultdict
import pprint
import cv2
from numpy.core.multiarray import ndarray
from tqdm import tqdm

matplotlib.use('TKAgg')


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
        for row in tqdm(bbox_description):
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
        for row in tqdm(bbox_description):
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







class_dict = class_dict()
Image_dict = Image_dict(class_dict, './annotations-human-bbox/train-annotations-bbox.csv')
text_dict = Text_dict('result170_1201.csv', windowsize('retult_size.csv'))

object = 'poster'
text = 'the'

keys = []
obj_bb_list = []
txt_bb_list = []
for key in tqdm(text_dict.keys()):
    obj_bb_list = [k['classname'].lower() for k in Image_dict[key]]
    txt_bb_list = [k['text'].lower() for k in text_dict[key]]
    if object in obj_bb_list and text in txt_bb_list:
        keys.append(key)
with open('{}_{}_IDs.csv'.format(object,text), 'w') as file:
   writer = csv.writer(file, lineterminator='\n')
   writer.writerow(keys)

