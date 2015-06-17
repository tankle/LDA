# -*- coding: utf-8 -*-
__author__ = 'tan'

import random

class LDA(object):
    '''
    Gibbs LDA
    '''

    def __init__(self, topics=10, alpha=0.1, beta=0.1, iter_num=100, top_words=10, random_state=None):

        self.T = topics
        self.alpha = alpha
        self.beta = beta
        self.iter_num = iter_num
        self.top_words = top_words

        self.dataset = []
        self.p = []        # T double类型，存储采样的临时变量
        self.Z = []        # M*doc.size()，文档中词的主题分布
        self.nw = []       # T*V，词i在主题j上的分布
        self.nwsum = []    # T，属于主题i的总词数
        self.nd = []       # M*T，文章i属于主题j的词个数
        self.ndsum = []    # M，文章i的词个数
        self.theta = []    # 文档-主题分布
        self.phi = []      # 主题-词分布

        if random_state is not None:
            random.seed(random_state)

    def fit(self, dataset):

        self.dataset = dataset
        self.init()

        print('Sampling %d iterations!' % self.iter_num)
        for x in xrange(self.iter_num):
            print 'Iteration %d ...' % (x+1)
            for i in xrange(len(self.dataset.docs)):
                for j in xrange(self.dataset.docs[i].length):
                    topic = self.gibbs_sample(i, j)
                    self.Z[i][j] = topic

        #
        self.compute_theta()
        self.compute_phi()


    def init(self):
        '''
        初始化参数
        :return:
        '''
        # 一维数组可以用下面简单的方式初始化，但二维以上就不行
        self.p = [0.0] * self.T
        self.nw = [[0] * self.dataset.V for y in xrange(self.T)]
        self.nwsum = [0] * self.T
        self.nd = [[0] * self.T for y in xrange(self.dataset.M)]
        self.ndsum = [0] * self.dataset.M
        self.theta = [[0.0] * self.T for y in xrange(self.dataset.M)]
        self.phi = [[0.0] * self.dataset.V for y in xrange(self.T)]

        self.Z = [[] for x in xrange(self.dataset.M)]
        for x in xrange(self.dataset.M):
            self.Z[x] = [0 for y in xrange(self.dataset.docs[x].length)]
            self.ndsum[x] = self.dataset.docs[x].length
            # 每个词的主题采用随机初始化方式
            for y in xrange(self.dataset.docs[x].length):
                # 随机产生0 到 self.T-1 之间的数字，包含self.T - 1
                topic = random.randint(0, self.T - 1)
                self.Z[x][y] = topic
                # try:
                self.nw[topic][self.dataset.docs[x].words[y]] += 1
                # except IndexError:
                #     print(len(self.nw))
                #     print(len(self.nw[0]))
                #     print(self.dataset.docs[x].words[y])
                #     print(str(topic))
                #     print(str(x))
                #     exit(1)
                self.nd[x][topic] += 1
                self.nwsum[topic] += 1


    def gibbs_sample(self, i, j):
        '''

        :param i:
        :param j:
        :return:
        '''
        topic = self.Z[i][j]
        wid = self.dataset.docs[i].words[j]
        self.nwsum[topic] -= 1
        self.nw[topic][wid] -= 1
        self.nd[i][topic] -= 1
        self.ndsum[i] -= 1

        Vbeta = self.dataset.V * self.beta
        Talpha = self.T * self.alpha

        for k in xrange(self.T):
            self.p[k] = (self.nw[k][wid] + self.beta)/(self.nwsum[k] + Vbeta) *\
                        (self.nd[i][k] + self.alpha)/(self.ndsum[i] + Talpha)

        for k in xrange(1, self.T):
            self.p[k] += self.p[k-1]

        u = random.uniform(0, self.p[self.T - 1])
        for topic in xrange(self.T):
            if self.p[topic] > u:
                break
        self.nwsum[topic] += 1
        self.nw[topic][wid] += 1
        self.nd[i][topic] += 1
        self.ndsum[i] += 1
        return topic



    def compute_theta(self):
        """
        doc-topic 矩阵
        :return:
        """
        for i in xrange(self.dataset.M):
            for j in xrange(self.T):
                self.theta[i][j] = (self.nd[i][j] + self.alpha) / \
                                   (self.ndsum[i] + self.alpha * self.T)

    def compute_phi(self):
        """
        topic-word 矩阵
        :return:
        """
        for i in xrange(self.T):
            for j in xrange(self.dataset.V):
                self.phi[i][j] = (self.nw[i][j] + self.beta) / \
                                 (self.nwsum[i] + self.beta * self.dataset.V)

    def predict(self, doc):
        p = [0.0] * self.T
        nw = [[0] * self.dataset.V for w in xrange(self.T)]
        nwsum = [0] * self.T
        nd = [0] * self.T
        theta = [0.0] * self.T

        Z = [0] * doc.length
        ndsum = doc.length
        # 采用随机初始化方式
        for w in xrange(doc.length):
            # 随机产生0 到 self.T-1 之间的数字，包含self.T - 1
            topic = random.randint(0, self.T - 1)
            Z[w] = topic
            nw[topic][doc.words[w]] += 1
            nd[topic] += 1
            nwsum[topic] += 1

        for i in xrange(self.iter_num):
            for j in xrange(doc.length):
                topic = Z[j]
                wid = doc.words[j]
                nwsum[topic] -= 1
                nw[topic][wid] -= 1
                nd[topic] -= 1
                ndsum -= 1

                Vbeta = self.dataset.V * self.beta
                Talpha = self.T * self.alpha

                for k in xrange(self.T):
                    p[k] = (nw[k][wid] + self.beta)/(nwsum[k] + Vbeta) *\
                           (nd[k] + self.alpha)/(ndsum + Talpha)

                for k in xrange(1, self.T):
                    p[k] += p[k-1]

                u = random.uniform(0, p[self.T - 1])
                for topic in xrange(self.T):
                    if p[topic] > u:
                        break
                nwsum[topic] += 1
                nw[topic][wid] += 1
                nd[topic] += 1
                ndsum += 1

                Z[j] = topic

        for i in xrange(self.T):
            theta[i] = (nd[i] + self.alpha) /\
                       (ndsum + self.alpha * self.T)

        return theta

    def save_model(self, file_prefix):
        with open(file_prefix+".theta", "w") as f:
            for i in xrange(self.dataset.M):
                for j in xrange(self.T):
                    f.write(str(self.theta[i][j]) + " ")
                f.write("\n")

        with open(file_prefix+".phi", "w") as f:
            for i in xrange(self.T):
                for j in xrange(self.dataset.V):
                    f.write(str(self.phi[i][j]) + " ")
                f.write("\n")

        with open(file_prefix+".topwords", "w") as f:
            if self.top_words > self.dataset.V:
                self.top_words = self.dataset.V

            for topic in xrange(self.T):
                f.write('Topic '+str(topic)+'th:\n')
                topic_words = []
                for y in xrange(self.dataset.V):
                    topic_words.append((y, self.phi[topic][y]))
                #quick-sort
                topic_words.sort(key=lambda x: x[1], reverse=True)
                for y in xrange(self.top_words):
                    word = self.dataset.id2word[topic_words[y][0]]
                    f.write('\t'+word+'\t'+str(topic_words[y][1])+'\n')

        with open(file_prefix+'.others', 'w') as fothers:
            fothers.write('alpha = '+str(self.alpha)+'\n')
            fothers.write('beta = '+str(self.beta)+'\n')
            fothers.write('ntopics = '+str(self.T)+'\n')
            fothers.write('ndocs = '+str(self.dataset.M)+'\n')
            fothers.write('nwords = '+str(self.dataset.V)+'\n')
            fothers.write('liter = '+str(self.iter_num)+'\n')
        pass


    def get_result(self):
        print("----doc-topic matrix result----")
        for i in xrange(self.dataset.M):
            print "doc: %d topic: " % i,
            topics = []
            for y in xrange(self.T):
                topics.append((y, self.theta[i][y]))
            #quick-sort
            topics.sort(key=lambda x: x[1], reverse=True)
            for j in xrange(self.T):
                print topics[j][0], ":", str(topics[j][1])+" ",
            print("")


        print("\n\n----topic-word matrix result----")
        if self.top_words > self.dataset.V:
            self.top_words = self.dataset.V

        for i in xrange(self.T):
            print "topic: %d words: " % i,
            topic_words = []
            for y in xrange(self.dataset.V):
                topic_words.append((y, self.phi[i][y]))
            #quick-sort
            topic_words.sort(key=lambda x: x[1], reverse=True)

            for j in xrange(self.top_words):
                print self.dataset.id2word[topic_words[j][0]], ":", (topic_words[j][1]), " ",
            print("")