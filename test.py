# -*- coding: utf-8 -*-
__author__ = 'tan'


from document import DataSet, Document
from ldaModel import LDA




if __name__ == "__main__":
    dataset = DataSet()

    infile = "data/dataset.txt"
    dataset.load(infile)
    # dataset.save_vocabulary("data/voc.txt")

    lda = LDA(topics=3, random_state=123)

    lda.fit(dataset)

    doc = Document()
    # a b c d a a a a
    doc.words = [0, 1, 2, 3, 0, 0, 0, 0]
    doc.length = len(doc.words)

    # theta = lda.theta
    # phi = lda.phi

    lda.get_result()

    theta = lda.predict(doc)
    print("----doc-topic matrix result----")
    for j in xrange(lda.T):
        print "topic: ", j, ":", str(theta[j])+" ",
    print("\n")
