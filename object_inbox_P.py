import numpy as np
import matplotlib
from typing import Any, List
from gensim.models import word2vec, Word2Vec
from nltk.corpus import stopwords
import csv
from collections import defaultdict
from numpy.core.multiarray import ndarray

matplotlib.use('TKAgg')


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