from pyspark.mllib.regression import LabeledPoint

import SetsCreation
from Classifiers import DecisionTree, RandomForest, RandomForest_sklearn, GradientBoostedTree, LogisticRegression, LinearSVC


def getBestResults(num_classifiers):
    with open('CSV Results/results1_final.csv', 'r') as resultsReader:
        best_result_lines = []

        # Ottengo i parametri del miglior risultato dei test di ogni classificatore
        for i in range(num_classifiers):

            # Calcolo il numero dei test eseguiti per questo classificatore
            line = resultsReader.readline()
            line = line.split(": ")
            classifier_name = line[0]
            nums_test = int(line[1].split(" ")[0])

            resultsReader.readline()  # salto l'header

            lines = []

            # Colleziono le righe dei test in un array
            for j in range(nums_test):
                lines.append(resultsReader.readline())

            # Trovo l'indice del test migliore
            line = resultsReader.readline().split(" ")
            index = int(line[len(line) - 1].split("°")[1])

            # Trovo il numero dei parametri usati per questo classificatore
            num_parameters = int(resultsReader.readline().split(" ")[1])

            # Trovo i parametri utilizzati
            parameters = []
            for j in range(num_parameters):
                parameters.append(resultsReader.readline().split(": ")[1][:-1])

            # Inserisco il tutto in un array
            best_result_lines.append((classifier_name, index, parameters, lines[index]))

            # Salto 10 righe
            for j in range(10):
                resultsReader.readline()

        return best_result_lines

def getLabelsAndPredictions(best_result_lines):

    datas = SetsCreation.setsCreation(1, 2)

    (trainingData, testData) = datas[0]
    c_trainingData = trainingData.map(lambda x: LabeledPoint(x[30], x[:30]))
    c_testData = testData.map(lambda x: LabeledPoint(x[30], x[:30]))

    labelsAndPredictions = {}

    for i in range(len(best_result_lines)):
        row = best_result_lines[i]
        parameters = row[2]
        if i is 0:
            labelsAndPredictions.update({row[0]: DecisionTree.decisionTree(c_trainingData, c_testData, parameters[0], int(parameters[1]), int(parameters[2])).collect()})
            print("1/6")
        elif i is 1:
            labelsAndPredictions.update({row[0]:RandomForest.randomForest(c_trainingData, c_testData, parameters[0], int(parameters[1]), int(parameters[2]), int(parameters[3])).collect()})
            print("2/6")
        elif i is 2:
            labelsAndPredictions.update({row[0]:RandomForest_sklearn.randomForest(c_trainingData, c_testData, int(parameters[1]), parameters[0], int(parameters[2]), parameters[3]).collect()})
            print("3/6")
        elif i is 3:
            labelsAndPredictions.update({row[0]:GradientBoostedTree.gradientBoostedTrees(c_trainingData, c_testData, parameters[0], int(parameters[3]), int(parameters[1]), int(parameters[2])).collect()})
            print("4/6")
        elif i is 4:
            labelsAndPredictions.update({row[0]:LogisticRegression.logisticRegression(trainingData, testData, int(parameters[0]), float(parameters[1]), float(parameters[2]), int(parameters[3])).collect()})
            print("5/6")
        elif i is 5:
            labelsAndPredictions.update({row[0]:LinearSVC.linearSVC(trainingData, testData, int(parameters[0]), float(parameters[1]), int(parameters[2])).collect()})
            print("6/6")

    return labelsAndPredictions


def esembler(labelsAndPredictions):

    esemblePair = []

    esemblePair.append({'DT RF': majorityVotePairs(labelsAndPredictions['Decision_Tree MLLib'], labelsAndPredictions['Random_Forest MLLib'])})
    esemblePair.append({'DT GBT': majorityVotePairs(labelsAndPredictions['Decision_Tree MLLib'], labelsAndPredictions['Gradient Boosted Tree'])})
    esemblePair.append({'DT LR': majorityVotePairs(labelsAndPredictions['Decision_Tree MLLib'], labelsAndPredictions['Logistic Regression'])})
    esemblePair.append({'DT LSVC': majorityVotePairs(labelsAndPredictions['Decision_Tree MLLib'], labelsAndPredictions['LinearSVC'])})
    esemblePair.append({'RF GBT': majorityVotePairs(labelsAndPredictions['Random_Forest MLLib'], labelsAndPredictions['Gradient Boosted Tree'])})
    esemblePair.append({'RF LR': majorityVotePairs(labelsAndPredictions['Random_Forest MLLib'], labelsAndPredictions['Logistic Regression'])})
    esemblePair.append({'RF LSVC': majorityVotePairs(labelsAndPredictions['Random_Forest MLLib'], labelsAndPredictions['LinearSVC'])})
    esemblePair.append({'GBT LR': majorityVotePairs(labelsAndPredictions['Gradient Boosted Tree'], labelsAndPredictions['Logistic Regression'])})
    esemblePair.append({'GBT LSVC': majorityVotePairs(labelsAndPredictions['Gradient Boosted Tree'], labelsAndPredictions['LinearSVC'])})
    esemblePair.append({'LR LSVC': majorityVotePairs(labelsAndPredictions['Logistic Regression'], labelsAndPredictions['LinearSVC'])})

    x = 0


def majorityVotePairs(one, two):

    if(len(one) != len(two)):
        return ArithmeticError

    onetwo = []

    for i in range(len(one)):
        if(one[i][0] == two[i][0]):
            onetwo.append(one[i])
        elif(one[i][0] == 1.0):
            onetwo.append(one[i])
        else:
            onetwo.append(two[i])

    return onetwo

def majorityVoteTriple(one, two, three):
    pass

def majorityVoteQuadruple(one, two, three, four):
    pass

def majorityVoteQuintuple(one, two, three, four, five):
    pass


if __name__ == '__main__':
    bestResults = getBestResults(num_classifiers=6)
    labelsAndPredictions = getLabelsAndPredictions(bestResults)
    boh = esembler(labelsAndPredictions)
