import numpy as np
import pandas as pd

import Perceptron2

def initiateTheArrays(fileName):
    dFrame = pd.read_csv(fileName, sep=';', header=None, decimal=',')
    rows, columns = dFrame.shape[0], dFrame.shape[1]
    xFromFile = np.asarray([[1.0 for j in range(columns)] for i in range(rows)])
    correctResultsFromFile = np.asarray([0 for i in range(rows)])

    for i in range(rows):
        for j in range(columns - 1):
            xFromFile[i][1 + j] = dFrame.iat[i, j]

    for i in range(rows):
        correctResultsFromFile[i] = dFrame.iat[i, columns - 1]

    return xFromFile, correctResultsFromFile, rows

eta = 0.1

def createNetwork(fileName):

    xSignals, correctResults, rows = initiateTheArrays(fileName)
    firstLayer = []

    for i in range(2):
        weights = np.asarray(np.random.rand(3))*2 - 1
        firstLayer.append(Perceptron2.Perceptron(xSignals[0], weights, eta))

    weights = np.asarray(np.random.rand(3))*2 - 1
    secondLayer = [Perceptron2.Perceptron(xSignals[0], weights, eta) for i in range(1)]

    return firstLayer, secondLayer, rows, xSignals, correctResults

def network(firstLayer, secondLayer, xSignals , correctResults, numberOfRows):
    responsesFromFirstLayer = np.asarray([1.0 for j in range(len(firstLayer) + 1)])

    errorValue = 0.0

    for i in range(numberOfRows):
        for j in range(len(firstLayer)):
            firstLayer[j].setSignals(xSignals[i])
            responsesFromFirstLayer[1 + j] = firstLayer[j].firstLayerResponse()

        secondLayer[0].setSignals(responsesFromFirstLayer)
        responseFromSecondLayer = secondLayer[0].secondLayerResponse()

        error = 0.5*(correctResults[i] - responseFromSecondLayer)

        print(xSignals[i][1], xSignals[i][2], responseFromSecondLayer, correctResults[i] , error)

        for j in range(len(firstLayer)):
            firstLayer[j].firstLayerWeightsCalculation(error, responseFromSecondLayer, secondLayer[0].weights[j+1], secondLayer[0].signals[j + 1])

        secondLayer[0].secondLayerWeightsCalculation(error, responseFromSecondLayer)

    for i in range(numberOfRows):
        for j in range(len(firstLayer)):
            firstLayer[j].setSignals(xSignals[i])
            responsesFromFirstLayer[1 + j] = firstLayer[j].firstLayerResponse()

        secondLayer[0].setSignals(responsesFromFirstLayer)

        responseFromSecondLayer = secondLayer[0].secondLayerResponse()

        error = 0.5*(correctResults[i] - responseFromSecondLayer)
        errorValue += error**2

    print(errorValue)
    return False

def learn(maxIter):
    epoch = 0
    print('Create:')
    firstLayer, secondLayer, numberOfRows, xSignals, correctResults = createNetwork('XOR.csv')

    while maxIter > epoch:

        isLearnt = network(firstLayer, secondLayer, xSignals, correctResults, numberOfRows)
        epoch += 1

        if isLearnt:
            print("Nauczony na fest po " + str(epoch) + " epokach")
            break
        else:
            pass
            #print("Minela epoka: " + str(epoch))

learn(1000)