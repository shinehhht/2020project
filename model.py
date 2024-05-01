import numpy as np
from bpmf import BPMF
import matplotlib.pyplot as plt
from math import *

def loadData(N, M, filename, min=0, max=5.5, sep=","):
    R = np.zeros((N, M), dtype=float)
    I = np.zeros((N, M))
    movieID = {}
    nextID = 0
    f = open(filename, "r")
    for line in f:
        if line[0] == 'u':
            # this is a comment
            continue
        user, movie, ranking, timestamp = line.split(sep)
        if movie not in movieID:
            if nextID >= M:
                continue
            movieID[movie] = nextID
            nextID += 1
        sigma = (float(ranking) - min) / (max-min)
        R[int(user) - 1, int(movieID[movie])] = np.log(sigma)-np.log(1-sigma)
        I[int(user) - 1, int(movieID[movie])] = 1
    return R, I, movieID

def evaluate(R_true, R_predict, I):
    return np.sum(np.absolute(R_true-R_predict)*I) / np.sum(I)

def sigmoid(z):
    return 1/(1 + np.exp(-z))

class BMF():
    def __init__(self, N, M, K, sigma=2.5, sigmaU=1, sigmaV=1, type='map'):
        self.U = np.random.normal(0, sigmaU, (N, K))
        self.V = np.random.normal(0, sigmaV, (M, K))
        self.sigma = sigma
        self.sigmaU = sigmaU
        self.sigmaV = sigmaV
        self.lambdaU = lambda: self.sigma**2 / self.sigmaU**2
        self.lambdaV = lambda: self.sigma**2 / self.sigmaV**2
        self.type = type
        self.mU = np.zeros((N, K))
        self.mV = np.zeros((M, K))

    def E(self, R, I):
        return 0.5*np.sum(I*np.square(R-self.U @ self.V.T))+0.5*self.lambdaU()*np.sum(np.square(self.U))+0.5*self.lambdaV()*np.sum(np.square(self.V))

    def SquareLoss(self, R_true, R_predict, I):
        return sqrt(np.sum(I*np.square(R_true-R_predict)) / np.sum(I))

    def gradient(self, R, I):
        N, _ = self.U.shape
        M, K = self.V.shape
        gU, gV = np.zeros_like(self.U), np.zeros_like(self.V)
        for n in range(N):
            sum = np.zeros(K)
            for m in range(M):
                sum += I[n, m] * (-1 * R[n, m] * self.V[m] + np.dot(self.U[n], self.V[m]) * self.V[m])
            gU[n] = (sum + self.lambdaU() * self.U[n])
        for m in range(M):
            sum = np.zeros(K)
            for n in range(N):
                sum += I[n, m] * (-1 * R[n, m] * self.U[n] + np.dot(self.U[n], self.V[m]) * self.U[n])
            gV[m] = (sum + self.lambdaV() * self.V[m])
        return gU, gV

    def optim(self, gU, gV, lr, method='SGD'):
        if method=='SGD':
            self.U -= gU * lr
            self.V -= gV * lr
        elif method=='momentum':
            self.mU -= gU * lr
            self.mV -= gV * lr
            self.U += self.mU
            self.V += self.mV

    def train(self, R, I, nEpoch = 20, lr=1e-3):
        MAP_Loss = [self.SquareLoss(sigmoid(R), self.predict(), I)]
        for _ in range(nEpoch):
            gU, gV = self.gradient(R, I)
            self.optim(gU, gV, lr, method='SGD')
            MAP_Loss.append(self.SquareLoss(sigmoid(R), self.predict(), I))
            print(f"epoch {_}:\n"
                  f"E={self.E(R, I)}\n"
                  f"RMSE={MAP_Loss[-1]}\n"
                  f"average difference of ground truth and predicted ranking: {evaluate(sigmoid(R), self.predict(), I)}")
        plt.plot(np.arange(nEpoch+1), np.array(MAP_Loss))
        plt.title('MAP loss curve')
        plt.xlabel('Eqoch')
        plt.ylabel('Square loss')
        plt.savefig('MAP loss curve.png')
        plt.show()
        #BPMF(R, R, self.U.T, self.V.T, 30, self.U.shape[1], 0, 0, 5.5)

    def predict(self):
        if self.type=='map':
            return sigmoid(self.U @ self.V.T)
        elif self.type=='BPMF':
            return 0

if __name__ == "__main__":
    N, M = 610, 9742
    R, I, movieID = loadData(N, M, 'dataset/ratings.csv')
    model = BMF(N, M, 5)
    model.train(R, I)
    R_predict = model.predict()
    print(R_predict, evaluate(sigmoid(R), R_predict, I))

