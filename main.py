import csv
from math import log2
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import os

NUM_BINS = 5
TREE_DEPTH = 3

def readData(path: str) -> list:
    with open(path, newline='') as f:
        reader = csv.reader(f)
        data = list(reader)
    return data

def makeBins(dataSet: list, numBins: int) -> list:
    binRanges = []
    # Loops through the columns in your dataset not counting the class label.
    for i in range(len(dataSet[0]) - 1):
        maxValue = max([float(num[i]) for num in dataSet])
        minValue = min([float(num[i]) for num in dataSet])

        binWidth = (maxValue - minValue) / numBins # Calculates the width of each bin
        binFeatureRange = []

        for k in range(numBins):
            binFeatureRange.append([minValue, (minValue + binWidth)])
            minValue += binWidth
        binRanges.append(binFeatureRange)

    return binRanges

def discretize(dataSet: list, binRanges: list) -> list:
    for i in range(len(dataSet)):
        for j in range(len(binRanges)):
            for k in range(len(binRanges[j])):
                if (float(dataSet[i][j]) >= binRanges[j][k][0] and float(dataSet[i][j]) < binRanges[j][k][1]):
                    dataSet[i][j] = k
                    break
                elif (k == 0 and float(dataSet[i][j]) <= binRanges[j][k][0]):
                    dataSet[i][j] = k
                    break
                elif (k == (len(binRanges[j])-1) and float(dataSet[i][j]) >= binRanges[j][k][1]):
                    dataSet[i][j] = k
                    break
    return dataSet

def calculateEntropy(dataSet: list) -> float:
    classLabels = {}
    entropy = 0.0
    for i in range(len(dataSet)):
        if dataSet[i][-1] in classLabels:
            classLabels[dataSet[i][-1]] += 1
        else:
            classLabels[dataSet[i][-1]] = 1
    totalClassLabels = sum(classLabels.values())
    for j in classLabels.values():
        entropy -= (j/totalClassLabels)*log2(j/totalClassLabels)
    return entropy

def calculateWeightedEntropy(dataSet: list, feature: int) -> list:
    totalFeatureEntropy = 0.0
    for i in range(len(np.unique([d for d in dataSet]))):
        featureClassLabels = {}
        featureTotal = 0.0
        entropy = 0
        for j in range(len(dataSet)):
            if dataSet[j][feature] == i:
                featureTotal += 1
                if dataSet[j][-1] in featureClassLabels:
                    featureClassLabels[dataSet[j][-1]] += 1
                else:
                    featureClassLabels[dataSet[j][-1]] = 1
        for j in featureClassLabels.values():
            entropy -= (j/featureTotal)*log2(j/featureTotal)
            
        entropy = entropy * float(len([items[feature] for items in dataSet if items[feature] == i])/len(dataSet))
        totalFeatureEntropy += entropy
    return totalFeatureEntropy
    
# Gain should be the entropy of the entire dataset - average entropy of the subsets we create
def calculateGain(dataSet: list, features) -> dict:
    gainMap = {}
    totalEntropy = calculateEntropy(dataSet)
    for i in features:
        featureEntropy = calculateWeightedEntropy(dataSet, i)
        gainMap[i] = totalEntropy - featureEntropy
    return gainMap


def maxGain(gainMap: dict) -> tuple:
    return max(gainMap.items(), key=lambda x: x[1])

def predict(query: dict, tree: dict) -> str:
    for key in list(query.keys()):
        if key in list(tree.keys()):
            try:
                result = tree[key][query[key]] 
            except:
                return 0
            result = tree[key][query[key]]
            if isinstance(result,dict):
                return predict(query,result)
            else:
                return result

def ID3(data: list, features: list, tree: dict, height: int) -> dict:
    # If the height of the tree has been met or there are no features left return the mode of the dataset class labels.
    if height == 0 or len(features) == 0:
        return np.unique([d[-1] for d in data])[np.argmax(np.unique([d[-1] for d in data], return_counts=True)[1])]

    # If there is only unique class labels return that label.
    if len(np.unique([d[-1] for d in data])) <= 1:
        return np.unique([d[-1] for d in data])[0]
    
    else:
        # Calculate the best feature to split on.
        best_feature = maxGain(calculateGain(data, features))[0]

        tree = {best_feature:{}}
        features = [i for i in features if i != best_feature]

        for value in np.unique([d[best_feature] for d in data]):
            # Create a subset of data split on the best feature by bin number.
            sub_data = [feature for feature in data if int(feature[best_feature]) == int(value)]

            # If the length of the subdata is 0 return the mode of the class labels
            if len(sub_data) == 0:
                return np.unique([d[-1] for d in data])[np.argmax(np.unique([d[-1] for d in data], return_counts=True)[1])]

            # Recursively call the function with the subdata.
            subtree = ID3(sub_data, deepcopy(features), tree, height-1)
            tree[best_feature][value] = subtree
        return tree

def predictData(data: list, tree: dict) -> float:
    total = 0
    for i in range(len(data)):
        x = predict({v:k for v, k in enumerate(data[i][:len(data[i])-1])}, tree)
        if x == data[i][-1]:
            total += 1
    return total/len(data) * 100


def printDecisionBoundries(data: list, binRanges: list, tree: dict, file: str):
    min1, max1 = min([float(x[0]) for x in data]), max([float(x[0]) for x in data])
    min2, max2 = min([float(x[1]) for x in data]), max([float(x[1]) for x in data])
    
    targetNames = ['0 Class Label', '1 Class Label']

    # Added 1 to the max and subtracted 1 from the min for better formatting of the graph.
    x1grid = np.arange(min1-1, max1+1, .1)
    x2grid = np.arange(min2-1, max2+1, .1)
    xx, yy = np.meshgrid(x1grid, x2grid)

    r1, r2 = xx.flatten(), yy.flatten()
    r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))
    grid = np.hstack((r1, r2))
    discretize(grid, binRanges)

    prediction = []
    for i in range(len(grid)):
        x = predict({v:k for v, k in enumerate(grid[i][:len(grid[i])])}, tree)
        prediction.append(x)

    zz = np.asarray(prediction).reshape(xx.shape)
    plt.contourf(xx, yy, zz, cmap="RdBu")

    for index, class_val in enumerate(range(2)):
        xvals = [float(x[0]) for x in data if int(x[-1]) == class_val]
        yvals = [float(x[1]) for x in data if int(x[-1]) == class_val]
        plt.scatter(xvals, yvals, label=targetNames[index], cmap="RdBu")
    plt.legend(loc="lower right")
    plt.xlabel("X Values")
    plt.ylabel("Y Value")
    plt.title(file + " dataset")
    plt.show()

if __name__ == "__main__":
    datasets = ["synthetic-1", "synthetic-2", "synthetic-3", "synthetic-4", "pokemon"]

    for file in datasets[:-1]:
        data = readData(os.getcwd()+'/data/'+file+'.csv')
        binRanges = makeBins(data, NUM_BINS)
        discretizedData = discretize(deepcopy(data), binRanges)
        decisionTree = ID3(discretizedData, [i for i in range(len(discretizedData[0])-1)], None, TREE_DEPTH-1)
        print(f"Predicted with an accuracy of {round(predictData(discretizedData, decisionTree), 2)}% for {file} dataset.")

        printDecisionBoundries(data, binRanges, decisionTree, file)

    data = readData(os.getcwd()+'/data/'+datasets[-1]+'.csv')
    binRanges = makeBins(data, NUM_BINS)
    discretizedData = discretize(deepcopy(data), binRanges)
    decisionTree = ID3(discretizedData, [i for i in range(len(discretizedData[0])-1)], None, TREE_DEPTH-1)
    print(f"Predicted with an accuracy of {round(predictData(discretizedData, decisionTree), 2)}% for {datasets[-1]} dataset.")