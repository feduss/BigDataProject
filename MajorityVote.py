# coding=utf-8
import csv
from pathlib import Path
from pyspark.shell import sc
import MetricsEvaluation as ma
import SetsCreation
from Classifiers import DecisionTree, RandomForest, GradientBoostedTree, LogisticRegression, LinearSVC


def getBestResults(source_file, num_classifiers):
    with open(str(Path(__file__).parent) + '/CSV_Results/' + source_file + '.csv', 'r') as resultsReader:
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


def getLabelsAndPredictions(best_result_lines, destination_file, used_dataset):
    datas = SetsCreation.setsCreation(1, used_dataset)

    (trainingData, testData) = datas[0]
    labelsAndPredictions = {}

    with open(str(Path(__file__).parent) + "/CSV_Results/" + destination_file + ".csv", "w") as ensemble_metric:
        csvWriter = csv.writer(ensemble_metric)

        csvWriter.writerow(['EnsembleType', 'Sensitivity', 'Fallout', 'Specificity', 'Miss_Rate', 'Test_Err', 'AUC'])

        for i in range(len(best_result_lines)):
            row = best_result_lines[i]
            parameters = row[2]
            if i is 0:
                labelsAndPredictions.update({row[0]: DecisionTree.decisionTree(trainingData,
                                                                               testData,
                                                                               parameters[0],
                                                                               int(parameters[1]),
                                                                               int(parameters[2]), True).collect()})
                print("1/5")
            elif i is 1:
                labelsAndPredictions.update({row[0]: RandomForest.randomForest(trainingData,
                                                                               testData,
                                                                               parameters[0],
                                                                               int(parameters[1]),
                                                                               int(parameters[2]),
                                                                               int(parameters[3]),
                                                                               True).collect()})
                print("2/5")
            elif i is 2:
                labelsAndPredictions.update({row[0]: GradientBoostedTree.gradientBoostedTrees(trainingData,
                                                                                              testData,
                                                                                              int(parameters[2]),
                                                                                              int(parameters[0]),
                                                                                              int(parameters[1]),
                                                                                              True).collect()})
                print("3/5")
            elif i is 3:
                labelsAndPredictions.update({row[0]: LogisticRegression.logisticRegression(trainingData,
                                                                                           testData,
                                                                                           int(parameters[0]),
                                                                                           float(parameters[1]),
                                                                                           float(parameters[2]),
                                                                                           int(parameters[3]),
                                                                                           True).collect()})
                print("4/5")
            elif i is 4:
                labelsAndPredictions.update({row[0]: LinearSVC.linearSVC(trainingData,
                                                                         testData,
                                                                         int(parameters[0]),
                                                                         float(parameters[1]),
                                                                         int(parameters[2]),
                                                                         True).collect()})
                print("5/5")

        for key in list(labelsAndPredictions.keys()):
            result = ma.metricsEvalutation(sc.parallelize(labelsAndPredictions[key]), len(labelsAndPredictions[key]), False)
            classifier_name = ""

            if key == "Decision_Tree":
                classifier_name = "DT"
            elif key == "Random_Forest":
                classifier_name = "RF"
            elif key == "Gradient_Boosted_Tree":
                classifier_name = "GBT"
            elif key == "Logistic_Regression":
                classifier_name = "LR"
            elif key == "Linear_SVC":
                classifier_name = "LSVC"

            csvWriter.writerow([classifier_name, str(result.sensitivity), str(result.fallout),
                                str(result.specificity), str(result.missRate),
                                str(result.testErr), str(result.AUC)])

    return labelsAndPredictions


def ensembler(predALab, destination_file):
    ensemblePair = {}
    '''
    dt = predALab['Decision_Tree']
    print("\nDT ("+str(len(dt)) +"): \n1) " +  str(dt[0]) + "\n2) " +  str(dt[1]) + "\n3) " +  str(dt[2]))

    rf = predALab['Random_Forest']
    print("\nRF ("+str(len(rf)) +"): \n1) " +  str(rf[0]) + "\n2) " +  str(rf[1]) + "\n3) " +  str(rf[2]))

    gbt = predALab['Gradient_Boosted_Tree']
    print("\nGBT ("+str(len(gbt)) +"): \n1) " +  str(gbt[0]) + "\n2) " +  str(gbt[1]) + "\n3) " +  str(gbt[2]))

    lr = predALab['Logistic_Regression']
    print("\nLR ("+str(len(lr)) +"): \n1) " +  str(lr[0]) + "\n2) " +  str(lr[1]) + "\n3) " +  str(lr[2]))

    lsvc = predALab['Linear_SVC']
    print("\nLSVC ("+str(len(lsvc)) +"): \n1) " +  str(lsvc[0]) + "\n2) " +  str(lsvc[1]) + "\n3) " +  str(lsvc[2]))
    '''

    print("Esecuzione ensemble pairs")
    ensemblePair.update({'DT RF': majorityVotePairs(predALab['Decision_Tree'],
                                                    predALab['Random_Forest'])})
    ensemblePair.update({'DT GBT': majorityVotePairs(predALab['Decision_Tree'],
                                                     predALab['Gradient_Boosted_Tree'])})
    ensemblePair.update({'DT LR': majorityVotePairs(predALab['Decision_Tree'],
                                                    predALab['Logistic_Regression'])})
    ensemblePair.update({'DT LSVC': majorityVotePairs(predALab['Decision_Tree'],
                                                      predALab['Linear_SVC'])})
    ensemblePair.update({'RF GBT': majorityVotePairs(predALab['Random_Forest'],
                                                     predALab['Gradient_Boosted_Tree'])})
    ensemblePair.update({'RF LR': majorityVotePairs(predALab['Random_Forest'],
                                                    predALab['Logistic_Regression'])})
    ensemblePair.update({'RF LSVC': majorityVotePairs(predALab['Random_Forest'],
                                                      predALab['Linear_SVC'])})
    ensemblePair.update({'GBT LR': majorityVotePairs(predALab['Gradient_Boosted_Tree'],
                                                     predALab['Logistic_Regression'])})
    ensemblePair.update({'GBT LSVC': majorityVotePairs(predALab['Gradient_Boosted_Tree'],
                                                       predALab['Linear_SVC'])})
    ensemblePair.update({'LR LSVC': majorityVotePairs(predALab['Logistic_Regression'],
                                                      predALab['Linear_SVC'])})
    print("Ensemble pairs eseguiti")

    ensembleTriple = {}

    print("Esecuzione ensemble triple")
    ensembleTriple.update({'DT RF GBT': majorityVoteTriple(predALab['Decision_Tree'],
                                                           predALab['Random_Forest'],
                                                           predALab['Gradient_Boosted_Tree'])})
    ensembleTriple.update({'DT RF LR': majorityVoteTriple(predALab['Decision_Tree'],
                                                          predALab['Random_Forest'],
                                                          predALab['Logistic_Regression'])})
    ensembleTriple.update({'DT RF LSVC': majorityVoteTriple(predALab['Decision_Tree'],
                                                            predALab['Random_Forest'],
                                                            predALab['Linear_SVC'])})
    ensembleTriple.update({'DT GBT LR': majorityVoteTriple(predALab['Decision_Tree'],
                                                           predALab['Gradient_Boosted_Tree'],
                                                           predALab['Logistic_Regression'])})
    ensembleTriple.update({'DT GBT LSVC': majorityVoteTriple(predALab['Decision_Tree'],
                                                             predALab['Gradient_Boosted_Tree'],
                                                             predALab['Linear_SVC'])})
    ensembleTriple.update({'DT LR LSVC': majorityVoteTriple(predALab['Decision_Tree'],
                                                            predALab['Logistic_Regression'],
                                                            predALab['Linear_SVC'])})
    ensembleTriple.update({'RF GBT LR': majorityVoteTriple(predALab['Random_Forest'],
                                                           predALab['Gradient_Boosted_Tree'],
                                                           predALab['Logistic_Regression'])})
    ensembleTriple.update({'RF GBT LSVC': majorityVoteTriple(predALab['Random_Forest'],
                                                             predALab['Gradient_Boosted_Tree'],
                                                             predALab['Linear_SVC'])})
    ensembleTriple.update({'RF LR LSVC': majorityVoteTriple(predALab['Random_Forest'],
                                                            predALab['Logistic_Regression'],
                                                            predALab['Linear_SVC'])})
    ensembleTriple.update({'GBT LR LSVC': majorityVoteTriple(predALab['Gradient_Boosted_Tree'],
                                                             predALab['Logistic_Regression'],
                                                             predALab['Linear_SVC'])})
    print("Ensemble triple eseguiti")

    ensembleQuadruple = {}

    print("Esecuzione ensemble quadruple")
    ensembleQuadruple.update({'DT RF GBT LR': majorityVoteQuadruple(predALab['Decision_Tree'],
                                                                    predALab['Random_Forest'],
                                                                    predALab['Gradient_Boosted_Tree'],
                                                                    predALab['Logistic_Regression'])})
    ensembleQuadruple.update({'DT RF GBT LSVC': majorityVoteQuadruple(predALab['Decision_Tree'],
                                                                      predALab['Random_Forest'],
                                                                      predALab['Gradient_Boosted_Tree'],
                                                                      predALab['Linear_SVC'])})
    ensembleQuadruple.update({'DT RF LR LSVC': majorityVoteQuadruple(predALab['Decision_Tree'],
                                                                     predALab['Random_Forest'],
                                                                     predALab['Logistic_Regression'],
                                                                     predALab['Linear_SVC'])})
    ensembleQuadruple.update({'DT GBT LR LSVC': majorityVoteQuadruple(predALab['Decision_Tree'],
                                                                      predALab['Gradient_Boosted_Tree'],
                                                                      predALab['Logistic_Regression'],
                                                                      predALab['Linear_SVC'])})
    ensembleQuadruple.update({'RF GBT LR LSVC': majorityVoteQuadruple(predALab['Random_Forest'],
                                                                      predALab['Gradient_Boosted_Tree'],
                                                                      predALab['Logistic_Regression'],
                                                                      predALab['Linear_SVC'])})
    print("Ensemble quadruple eseguiti")

    ensembleQuintuple = {}

    print("Esecuzione ensemble quintuple")
    ensembleQuintuple.update({'DT RF GBT LR LSVC': majorityVoteQuintuple(predALab['Decision_Tree'],
                                                                         predALab['Random_Forest'],
                                                                         predALab['Gradient_Boosted_Tree'],
                                                                         predALab['Logistic_Regression'],
                                                                         predALab['Linear_SVC'])})
    print("Ensemble quintuple eseguito")

    result = {}

    print("########################")
    print("Pair: " + str(ensemblePair))
    print("Triple: " + str(ensembleTriple))
    print("Quadruple: " + str(ensembleQuadruple))
    print("Quintuple: " + str(ensembleQuintuple))
    print("########################")

    for i in range(len(list(ensemblePair.keys()))):
        result.update({list(ensemblePair.keys())[i]: ma.metricsEvalutation(sc.parallelize(list(ensemblePair.values())[i]), len(list(ensemblePair.values())[i]), False)})

    for i in range(len(list(ensembleTriple.keys()))):
        result.update({list(ensembleTriple.keys())[i]: ma.metricsEvalutation(sc.parallelize(list(ensembleTriple.values())[i]), len(list(ensembleTriple.values())[i]), False)})

    for i in range(len(list(ensembleQuadruple.keys()))):
        result.update({list(ensembleQuadruple.keys())[i]: ma.metricsEvalutation(sc.parallelize(list(ensembleQuadruple.values())[i]), len(list(ensembleQuadruple.values())[i]), False)})

    result.update({list(ensembleQuintuple.keys())[0]: ma.metricsEvalutation(sc.parallelize(list(ensembleQuintuple.values())[0]), len(list(ensembleQuintuple.values())[0]), False)})

    with open(str(Path(__file__).parent) + "/CSV_Results/" + destination_file + ".csv", "a") as ensemble_metric:
        csvWriter = csv.writer(ensemble_metric)

        # csvWriter.writerow(['EnsembleType', 'Sensitivity', 'Fallout', 'Specificity', 'Miss_Rate', 'Test_Err', 'AUC'])
        for i in range(len(list(result.keys()))):
            csvWriter.writerow([list(result.keys())[i],
                                str(list(result.values())[i].sensitivity), str(list(result.values())[i].fallout),
                                str(list(result.values())[i].specificity), str(list(result.values())[i].missRate),
                                str(list(result.values())[i].testErr), str(list(result.values())[i].AUC)])
        csvWriter.writerow(["##################"])
        csvWriter.writerow(['DT = Decision_Tree'])
        csvWriter.writerow(['RF = Random_Forest'])
        csvWriter.writerow(['GBT = Gradient_Boosted_Tree'])
        csvWriter.writerow(['LR = Logistic_Regression'])
        csvWriter.writerow(['LSVC = Linear_SVC'])


def majorityVotePairs(one, two):
    if len(one) != len(two):
        return ArithmeticError

    onetwo = []

    for i in range(len(one)):
        if one[i][0] == two[i][0]:
            onetwo.append(one[i])
        elif one[i][0] == 1.0:
            onetwo.append(one[i])
        else:
            onetwo.append(two[i])

    return onetwo


def majorityVoteTriple(one, two, three):
    if len(one) != len(two) or len(one) != len(three) or len(two) != len(three):
        return ArithmeticError

    onetwothree = []

    for i in range(len(one)):
        onei = one[i]
        twoi = two[i]
        threei = three[i]

        sum = onei[0] + twoi[0] + threei[0]

        if onei[1] == twoi[1] and onei[1] == threei[1]:
            # La maggior parte è fraudolenta (almeno 2/3)
            if sum == 3.0 or sum == 2.0:
                onetwothree.append((onei[1], 1.0))
            else:
                onetwothree.append((onei[1], 0.0))
        else:
            return AssertionError

    return onetwothree


def majorityVoteQuadruple(one, two, three, four):
    if len(one) != len(two) or len(one) != len(three) or len(one) != len(four) \
            or len(two) != len(three) or len(two) != len(four) \
            or len(three) != len(four):
        return ArithmeticError

    onetwothreefour = []

    for i in range(len(one)):
        onei = one[i]
        twoi = two[i]
        threei = three[i]
        fouri = four[i]

        sum = onei[0] + twoi[0] + threei[0] + fouri[0]

        if onei[1] == twoi[1] and onei[1] == threei[1] and onei[1] == fouri[1]:
            # La maggior parte è fraudolenta (almeno 3/4)
            if sum == 4.0 or sum == 3.0:
                onetwothreefour.append((onei[1], 1.0))
            else:
                onetwothreefour.append((onei[1], 0.0))
        else:
            return AssertionError

    return onetwothreefour


def majorityVoteQuintuple(one, two, three, four, five):
    if len(one) != len(two) or len(one) != len(three) or len(one) != len(four) or len(one) != len(five) \
            or len(two) != len(three) or len(two) != len(four) or len(two) != len(five) \
            or len(three) != len(four) or len(three) != len(five) \
            or len(four) != len(five):
        return ArithmeticError

    onetwothreefourfive = []

    for i in range(len(one)):
        onei = one[i]
        twoi = two[i]
        threei = three[i]
        fouri = four[i]
        fivei = five[i]

        sum = onei[0] + twoi[0] + threei[0] + fouri[0] + fivei[0]

        if onei[1] == twoi[1] and onei[1] == threei[1] and onei[1] == fouri[1] and onei[1] == fivei[1]:
            # La maggior parte è fraudolenta (almeno 3/5)
            if sum == 5.0 or sum == 4.0 or sum == 3.0:
                onetwothreefourfive.append((onei[1], 1.0))
            else:
                onetwothreefourfive.append((onei[1], 0.0))
        else:
            return AssertionError

    return onetwothreefourfive


'''
if __name__ == '__main__':
    ra.ResultAnalysis(5, 'classifiers_metrics1_final', 'results1_final')
    ra.ResultAnalysis(5, 'classifiers_metrics2_final', 'results2_final')

    for i in range(1, 3):
        bestResults = getBestResults(num_classifiers=5, used_dataset=i)
        predALab = getLabelsAndPredictions(bestResults, used_dataset=i)
        ensembler(predALab, used_dataset=i)
'''
