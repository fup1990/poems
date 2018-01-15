import collections
import math
import random
import zipfile
import os
import numpy as np
import urllib
import tensorflow as tf

url = 'http://mattmahoney.net/dc/'

# 下载文本数据，并核对文件尺寸
def maybe_download(filename, expected_bytes):
    if not os.path.exists(filename):
        filename, _ = urllib.request.urlretrieve(url + filename, filename)

    statinfo = os.stat(filename)

    if statinfo.st_size == expected_bytes:
        print('Found and verified', filename)
    else:
        print(statinfo.st_size)
        raise Exception('Failed to verify ' + filename + '. Can you get to it with a browser?')
    return filename

filename = maybe_download('text8.zip', 31344016)

# 解压文件，并将数据转成单词列表
def read_data(filename):
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data

words = read_data(filename)
print('Data size', len(words))

# 定义词汇数量
vocabulary_size = 50000

def build_dataset(words):
    count = [['UNK', -1]]
    # 统计每个字的频数
    counter = collections.Counter(words)
    # 添加到list中
    count.extend(counter.most_common(vocabulary_size - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary

data, count, dictionary, reverse_dictionary = build_dataset(words)

del words

print('Most common words (+UNK', count[:5])
print('Simple data', data[:10], [reverse_dictionary[i] for i in data[:10]])






















