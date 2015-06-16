# -*- coding: utf-8 -*-
__author__ = 'tan'

# 单个文档
class Document(object):
    def __init__(self):
        self.words = []
        self.length = 0

# 整个文档集合
class DataSet(object):
    def __init__(self):
        self.M = 0
        self.V = 0
        self.docs = []
        self.word2id = {}
        self.id2word = {}

    # 从文件中加载文档,并转换为词汇
    def load(self, filename):
        with open(filename, "r") as f:
            print("Loading data from " + filename)
            lines = f.readlines()
            idx = 0
            docnum = 0
            for line in lines:
                line = line.strip()
                if len(line) == 0:
                    continue

                docnum += 1

                words = line.split()
                doc = Document()

                for word in words:
                    if word not in self.word2id:
                        self.word2id[word] = idx
                        self.id2word[idx] = word
                        doc.words.append(idx)
                        idx += 1
                    else:
                        doc.words.append(self.word2id[word])

                doc.length = len(words)
                self.docs.append(doc)

            self.M = docnum
            self.V = len(self.word2id)
            print('There are %d documents' % self.M)
            print('There are %d items' % self.V)


    # 保存词汇列表
    def save_vocabulary(self, filename):
        with open(filename, 'w') as f:
            for k, v in self.word2id.items():
                f.write(k + '\t' + str(v) + '\n')