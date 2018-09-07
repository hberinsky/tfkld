## weight.py
## Author: Yangfeng Ji
## Date: 09-06-2014
## Time-stamp: <yangfeng 09/08/2014 20:08:44>

from sklearn.feature_extraction.text import CountVectorizer
from cPickle import load, dump
import numpy, gzip
import scipy.sparse as ssp

class TFKLD(object):
    def __init__(self, ftrain, fdev, ftest):
        self.ftrain, self.fdev, self.ftest = ftrain, fdev, ftest
        self.trnM, self.trnL = None, None
        self.devM, self.devL = None, None
        self.tstM, self.tstL = None, None
        self.weight = None

    def loadtext(self, fname):
        text, label = [], []
        with open(fname, 'r') as fin:
            for line in fin:
                items = line.strip().split("\t")
                label.append(int(items[0]))
                text.append(items[1])
                text.append(items[2])
        return text, label


    def createdata(self):
        trnT, trnL = self.loadtext(self.ftrain)
        devT, devL = self.loadtext(self.fdev)
        tstT, tstL = self.loadtext(self.ftest)
        # Change the parameter setting in future
        countizer = CountVectorizer(dtype=numpy.float,
                                    ngram_range=(1,2))
        trnM = countizer.fit_transform(trnT)
        self.trnM, self.trnL = trnM, trnL
        devM = countizer.transform(devT)
        self.devM, self.devL = devM, devL
        tstM = countizer.transform(tstT)
        self.tstM, self.tstL = tstM, tstL
        self.trnM = ssp.csc_matrix(self.trnM)
        self.devM = ssp.csc_matrix(self.devM)
        self.tstM = ssp.csc_matrix(self.tstM)


    def weighting(self):
        print 'Create data matrix ...'
        self.createdata()
        print 'Counting features ...'
        M = self.trnM
        print 'type(M) = {}'.format(type(M))
        L = numpy.array(self.trnL)
        nRow, nDim = M.shape
        print 'nRow, nDim = {}, {}'.format(nRow, nDim)
        # (0, F), (0, T), (1, F), (1, T)
        count = numpy.ones((4, nDim))
        X = ssp.hstack((M[[i for i in xrange(nRow) if i % 2 == 0], :],
                        M[[i for i in xrange(nRow) if i % 2 == 1], :]))
        label_0, label_1 = (L == 0), (L == 1)
        for d in range(nDim):
            w1_d = (X[:, d] > 0).T.toarray()[0]
            w2_d = (X[:, d + nDim] > 0).T.toarray()[0]
            shared = w1_d & w2_d
            non_shared = w1_d ^ w2_d
            count[0,d] = 1 + sum(non_shared & label_0)
            count[2,d] = 1 + sum(non_shared & label_1)
            count[1,d] = 1 + sum(shared & label_0)
            count[3,d] = 1 + sum(shared & label_1)

        # Compute KLD
        print 'Compute KLD weights ...'
        weight = self.computeKLD(count)
        # Apply weights
        print 'Weighting ...'
        self.__weighting()


    def computeKLD(self, count):
        # Smoothing
        count = count + 0.05
        # Normalize
        pattern = [[1,1,0,0],[1,1,0,0],[0,0,1,1],[0,0,1,1]]
        pattern = numpy.array(pattern)
        prob = count / (pattern.dot(count))
        #
        ratio = numpy.log((prob[0:2,:] / prob[2:4,:]) + 1e-7)
        self.weight = (ratio * prob[0:2,:]).sum(axis=0)
        print self.weight.shape


    def __weighting(self):
        weight = ssp.lil_matrix(self.weight)
        print 'Applying weighting to training examples'
        self.trnM = self.trnM.multiply(weight)
        self.devM = self.devM.multiply(weight)
        self.tstM = self.tstM.multiply(weight)


    def save(self, fname):
        D = {'trnM':self.trnM, 'trnL':self.trnL,
             'devM':self.devM, 'devL':self.devL,
             'tstM':self.tstM, 'tstL':self.tstL}
        with gzip.open(fname, 'w') as fout:
            dump(D, fout)
        print 'Done'


def main():
    ftrain = "../data/train.data"
    fdev = "../data/dev.data"
    ftest = "../data/test.data"
    tfkld = TFKLD(ftrain, fdev, ftest)
    tfkld.weighting()
    # tfkld.createdata()
    tfkld.save("original-data.pickle.gz")


if __name__ == "__main__":
    main()


