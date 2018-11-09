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
from tqdm import tqdm
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import skimage.io as io

def class_dict():     #引数：なし　戻り値：OpenImagesのオブジェクトクラスとIDの辞書型の値（｛ID：クラス｝）
  class_dict = {}
  with open('class-descriptions.csv', newline='') as csvfile:
    class_description = csv.reader(csvfile, delimiter=',', quotechar='"')
    for row in class_description:
      class_dict[row[0]] = row[1]
  return class_dict



class_dict = class_dict()
with open('./annotations-human-bbox/train/annotations-human-bbox.csv', newline='') as csvfile:
  next(csvfile)
  bbox_description = csv.reader(csvfile, delimiter=',', quotechar='"')
  with open('annotations-human-bbox_train.csv', 'w') as f:
    writer = csv.writer(f, lineterminator='\n')  # 改行コード（\n）を指定しておく
    for row in tqdm(bbox_description):
      row[2] = class_dict[row[2]]
      writer.writerow(row)


