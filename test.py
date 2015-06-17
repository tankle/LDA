# -*- coding: utf-8 -*-
__author__ = 'tan'


from document import DataSet, Document
from ldaModel import LDA
import codecs

def testJieba():
    import jieba
    inname = "data/163news.txt"
    outname = "data/163news.split.txt"
    outfile = codecs.open(outname, "w", encoding="utf-8")
    print("spliting the dataset...")
    with open(inname, "r") as f:
        lines = f.readlines()
        for line in lines:
            seg_list = jieba.cut(line, cut_all=False)
            for word in seg_list:
                outfile.write(word+" ")
            outfile.write("")

    outfile.close()
    print("Over!!!")


def testChinese():
    dataset = DataSet()

    # infile = "data/dataset.txt"
    infile = "data/163news.split.txt"
    dataset.load(infile)
    dataset.save_vocabulary("data/voc.txt")

    lda = LDA(topics=10, random_state=123)

    lda.fit(dataset)

    # doc = Document()
    # # a b c d a a a a
    # doc.words = [0, 1, 2, 3, 0, 0, 0, 0]
    # doc.length = len(doc.words)

    # theta = lda.theta
    # phi = lda.phi

    lda.get_result()
    lda.save_model(file_prefix="data/163news")



def testLDA():
    dataset = DataSet()

    infile = "data/dataset.txt"

    dataset.load(infile)
    dataset.save_vocabulary("data/voc.txt")

    lda = LDA(topics=10, random_state=123)

    lda.fit(dataset)

    doc = Document()
    # a b c d a a a a
    doc.words = [0, 1, 2, 3, 0, 0, 0, 0]
    doc.length = len(doc.words)

    # theta = lda.theta
    # phi = lda.phi

    lda.get_result()
    lda.save_model(file_prefix="data/test")

    theta = lda.predict(doc)
    print("----doc-topic matrix result----")
    for j in xrange(lda.T):
        print "topic: ", j, ":", str(theta[j])+" ",
    print("\n")


if __name__ == "__main__":
    # testJieba()

    # testLDA()

    testChinese()