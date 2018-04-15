import math
import copy

def sigmoid(x, beta = 2.0):
    return 1.0 / (1.0 + (math.exp(-beta*x)))

def sigmoidDerivative(x, beta = 2.0):
    return beta*x*(1.0 - x)

def tanHip(x, beta = 2.0):
    return (1.0 - math.exp(-beta*x))/(1.0 + math.exp(-beta*x))

def tanHipDerivative(x, beta = 2.0):
    return beta*2.0*(1.0 - x**2)

class Perceptron:
    def __init__(self, signals, weights, eta):
        self.signals = copy.copy(signals)
        self.weights = copy.copy(weights)
        self.eta = eta

    def secondLayerResponse(self):
        r = self.weights[0]

        for i in range(1, len(self.weights)):
            r += self.signals[i] * self.weights[i]

        return sigmoid(r)

    def firstLayerResponse(self):
        r = self.weights[0]

        for i in range(1, len(self.weights)):
            r += self.signals[i] * self.weights[i]

        return tanHip(r)

    def secondLayerWeightsCalculation(self, error, response):
        for i in range(len(self.weights)):
            self.weights[i] += self.eta*error*sigmoidDerivative(response)*self.signals[i]

    def firstLayerWeightsCalculation(self, error, response, secondWeights_i, hSignals):
        for i in range(len(self.weights)):
            self.weights[i] += self.eta*error*sigmoidDerivative(response)*secondWeights_i*tanHipDerivative(hSignals)*self.signals[i]

    def setSignals(self, xSignals):
        self.signals = copy.copy(xSignals)

    def setWeights(self, weights):
        self.weights = copy.copy(weights)