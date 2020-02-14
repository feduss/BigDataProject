import csv
import sys

import SetsCreation
from pyspark.mllib.regression import LabeledPoint
import Classifier_DecisionTree as cdt
import Classifier_RandomForest_sklearn as crf_sk
import Classifier_RandomForest as crf
import Classifier_GradientBoostedTree as cgbt
import Classifier_MultilayerPerceptron as cmlp
import ResultAnalysis as ra

verbose = False  # Per stampare o meno i risultati di tutti i test
multiplier = 2   # Ripetizioni del singolo test

# File per testare diversi trainingset e testset sui classificatori implementati, fornendo diversi parametri
datas = SetsCreation.setsCreation(multiplier)

# DecisionTree (libreria MLLib) = DT
# RandomForest (libreria MLLib) = RFML
# RandomForest (libreria SkLearn) = RFSL
# GradientBoostedTree (libreria MLLib) = GBT
# Multilayer Perceptron (libreria ML.classification) = MLC

impurity = ['gini', 'entropy']  # DT, RFML, RFSL
maxDepth = [5, 6, 7]            # DT, RFML, RFSL
maxBins = [32, 64, 128]         # DT, RFML, GBT
numIterations = [50, 100]       # GBT, MLC

# Solo Decision Tree
minInstancesPerNode = [1]
minInfoGain = [0.0]

# Solo Random Forest MLLib
numTrees = [100, 200]

# Solo Random Forest sklearn
n_estimators = [100, 200]
max_features = ['auto', 'sqrt', 'log2']

# Solo Gradient Boosted
loss = ['logLoss', 'leastSquaresError', 'leastAbsoluteError']
maxDepth2 = [3, 5]

# Solo Multilayer Perceptron
layer = [[30, 10, 2], [30, 20, 2], [30, 20, 10, 2]]
blockSize = [128, 256, 512]
solver = ['l-bfgs', 'gd']

# Contatori
iter_count = 0
max_count = 1  # Da aggiornare ogni volta in ogni for
j = -1
k = -1
l = -1
m = -1

# Variabili gestione minore per tipo
index_min_err = -1
iter_min_err = -1
j_min_err = -1
k_min_err = -1
l_min_err = -1
m_min_err = -1
result_min = ra.Results(1, 1, 1, 1, 1, 1)

with open('classifiers_results.csv', 'w') as result_file:
    with open('best_classifiers_results.csv', 'w') as best_result_file:
        csvWriter = csv.writer(result_file)
        csvBestWriter = csv.writer(best_result_file)

        csvBestWriter.writerow(['Model', 'Index best test', 'Sensitivity', 'Fallout',
                                'Specificity', 'Miss_Rate', 'Test_Err', 'AUC'])

        csvWriter.writerow(['Decision_Tree MLLib'])
        csvWriter.writerow(['Iteration', 'Impurity', 'Max_Depth', 'Max_Bins',
                            'Sensitivity', 'Fallout', 'Specificity', 'Miss_rate',
                            'Test_Error', 'AUC'])

        # impurity = ['gini', 'entropy']
        # maxDepth = [5, 6, 7]
        # maxBins = [32, 64, 128]

        # DECISION TREE MLLib
        max_count = 18
        for i in range(0, 18):
            j = 0 if i < 9 else 1  # impurity
            k = int(i/3) % 3       # maxDepth
            l = i % 3              # maxBins

            for t in range(0, multiplier):
                if verbose:
                    print("\n--------------------------------------------------\n")
                    print("Test " + str(i+1) + "." + str(t+1)
                          + " con impurity: " + impurity[j] + ", maxDepth: " + str(maxDepth[k])
                          + ", maxBins:" + str(maxBins[l]) + ", minInstancesPerNode: 1, minInfoGain: 1 ")
                else:
                    iter_count = (i * multiplier) + t
                    percentage = int((iter_count / (max_count * multiplier)) * 100)
                    sys.stdout.write('\rPercentuale completamento test DT: ' + str(percentage) + "%"),
                    sys.stdout.flush()

                (trainingData, testData) = datas[t]
                c_trainingData = trainingData.map(lambda x: LabeledPoint(x[30], x[:30]))
                c_testData = trainingData.map(lambda x: LabeledPoint(x[30], x[:30]))
                testRecordsNumber = float(testData.count())

                labelsAndPredictions = cdt.decisionTree(c_trainingData, c_testData, impurity[j], maxDepth[k], maxBins[l],
                                                        minInstancesPerNode[0], minInfoGain[0])

                results = ra.resultAnalisys(labelsAndPredictions, testRecordsNumber, verbose)

                if results.testErr < result_min.testErr:
                    index_min_err = i
                    iter_min_err = t
                    j_min_err = j
                    k_min_err = k
                    l_min_err = l
                    result_min = results

                csvWriter.writerow([str(i+1) + "." + str(t+1),
                                    impurity[j], str(maxDepth[k]), str(maxBins[l]),
                                    str(results.sensitivity), str(results.fallout), str(results.specificity),
                                    str(results.missRate), str(results.testErr), str(results.AUC)])

            # input("Premi invio per continuare...\n")
            
        print("\nMiglior risultato DecisionTreeModel MLLib: test " + str(index_min_err+1) + "." + str(iter_min_err+1))
        print("Impurity: " + impurity[j_min_err] + ", maxDepth: " + str(maxDepth[k_min_err]) + ", maxBins:"
              + str(maxBins[l_min_err]) + ", minInstancesPerNode: 1, minInfoGain: 1 ")
        print("Tasso di errore: " + str(result_min.testErr * 100) + "%")
        print('--------------------------------------------------\n')

        csvBestWriter.writerow(['DecisionTreeModel', str(index_min_err+1) + "." + str(iter_min_err+1),
                                result_min.sensitivity, result_min.fallout, result_min.specificity, result_min.missRate,
                                result_min.testErr, result_min.AUC])

        csvWriter.writerow(['#############################'])

        j = -1
        k = -1
        l = -1
        index_min_err = -1
        iter_min_err = -1
        j_min_err = -1
        k_min_err = -1
        l_min_err = -1
        result_min = ra.Results(1, 1, 1, 1, 1, 1)
        iter_count = 0

        csvWriter.writerow(['Random_Forest MLLib'])
        csvWriter.writerow(['Iteration', 'Impurity', 'Max_Depth', 'Max_Bins', 'Num_Trees',
                            'Sensitivity', 'Fallout', 'Specificity', 'Miss_rate',
                            'Test_Error', 'AUC'])

        # impurity = ['gini', 'entropy']
        # maxDepth = [5, 6, 7]
        # maxBins = [32, 64, 128]
        # numTrees = [100, 200]

        # RANDOM FOREST MLLib
        max_count = 35
        for i in range(0, 35):
            j = 0 if i < 18 else 1  # impurity
            k = int(i / 6) % 3      # maxDepth
            l = int(i / 3) % 2      # maxBins
            m = i % 3               # numTrees

            for t in range(0, multiplier):
                if verbose:
                    print("\n--------------------------------------------------\n")
                    print("Test " + str(i+1) + "." + str(t+1)
                          + " con impurity: " + impurity[j] + ", maxDepth: " + str(maxDepth[k])
                          + ", maxBins: " + str(maxBins[m]) + ", numTrees: " + str(numTrees[l]))
                else:
                    iter_count = (i * multiplier) + t
                    percentage = int((iter_count / (max_count * multiplier)) * 100)
                    sys.stdout.write('\rPercentuale completamento test RFML: ' + str(percentage) + "%"),
                    sys.stdout.flush()

                (trainingData, testData) = datas[t]
                c_trainingData = trainingData.map(lambda x: LabeledPoint(x[30], x[:30]))
                c_testData = trainingData.map(lambda x: LabeledPoint(x[30], x[:30]))
                testRecordsNumber = float(testData.count())

                labelsAndPredictions = crf.randomForest(c_trainingData, c_testData, impurity[j], maxDepth[k], maxBins[m],
                                                        numTrees[l])

                results = ra.resultAnalisys(labelsAndPredictions, testRecordsNumber, verbose)

                if results.testErr < result_min.testErr:
                    index_min_err = i
                    iter_min_err = t
                    j_min_err = j
                    k_min_err = k
                    l_min_err = l
                    m_min_err = m
                    result_min = results

                csvWriter.writerow([str(i+1) + "." + str(t+1),
                                    impurity[j], str(maxDepth[k]), str(maxBins[m]), str(numTrees[l]),
                                    str(results.sensitivity), str(results.fallout), str(results.specificity),
                                    str(results.missRate), str(results.testErr), str(results.AUC)])

            # input("Premi invio per continuare...\n")

        # N.B. Può capitare che certi test ottengano lo stesso risultato, ma solo uno viene etichettato come migliore
        print("\nMiglior risultato RandomForestModel MLLib: test " + str(index_min_err+1) + "." + str(iter_min_err+1))
        print("Impurity: " + impurity[j_min_err] + ", maxDepth: " + str(maxDepth[k_min_err]) + ", maxBins: "
              + str(maxBins[m_min_err]) + ", numTrees: " + str(numTrees[l_min_err]))
        print("Tasso di errore: " + str(result_min.testErr * 100) + "%")
        print('--------------------------------------------------\n')

        csvBestWriter.writerow(['RandomForest MLLib', str(index_min_err+1) + "." + str(iter_min_err+1),
                                result_min.sensitivity, result_min.fallout, result_min.specificity, result_min.missRate,
                                result_min.testErr, result_min.AUC])

        csvWriter.writerow(['#############################'])

        j = -1
        k = -1
        l = -1
        m = -1
        index_min_err = -1
        iter_min_err = -1
        j_min_err = -1
        k_min_err = -1
        l_min_err = -1
        m_min_err = -1
        result_min = ra.Results(1, 1, 1, 1, 1, 1)
        iter_count = 0

        csvWriter.writerow(['Random_Forest Sklearn'])
        csvWriter.writerow(['Iteration', 'Impurity', 'Max_Depth', 'N_Estimators', 'Max_Features',
                            'Sensitivity', 'Fallout', 'Specificity', 'Miss_rate',
                            'Test_Error', 'AUC'])

        # impurity = ['gini', 'entropy']
        # maxDepth = [5, 6, 7]
        # n_estimators = [100, 200]
        # max_features = ['auto', 'sqrt', 'log2']

        # RANDOM FOREST Sklearn
        max_count = 35
        for i in range(0, 35):
            j = 0 if i < 18 else 1  # impurity
            k = int(i/6) % 3        # maxDepth
            l = int(i/3) % 2        # n_estimators
            m = i % 3               # max_features

            for t in range(0, multiplier):
                if verbose:
                    print("\n--------------------------------------------------\n")
                    print("Test " + str(i+1) + "." + str(t+1)
                          + " con impurity: " + impurity[j] + ", maxDepth: " + str(maxDepth[k])
                          + ", n_estimators: " + str(n_estimators[l]) + ", max_features: " + str(max_features[m]))
                else:
                    iter_count = (i * multiplier) + t
                    percentage = int((iter_count / (max_count * multiplier)) * 100)
                    sys.stdout.write('\rPercentuale completamento test RFSL: ' + str(percentage) + "%"),
                    sys.stdout.flush()

                (trainingData, testData) = datas[t]
                c_trainingData = trainingData.map(lambda x: LabeledPoint(x[30], x[:30]))
                c_testData = trainingData.map(lambda x: LabeledPoint(x[30], x[:30]))
                testRecordsNumber = float(testData.count())

                labelsAndPredictions = crf_sk.randomForest(c_trainingData, c_testData, n_estimators[l], impurity[j],
                                                           maxDepth[k], max_features[m])

                results = ra.resultAnalisys(labelsAndPredictions, testRecordsNumber, verbose)

                if results.testErr < result_min.testErr:
                    index_min_err = i
                    iter_min_err = t
                    j_min_err = j
                    k_min_err = k
                    l_min_err = l
                    m_min_err = m
                    result_min = results

                csvWriter.writerow([str(i+1) + "." + str(t+1),
                                    impurity[j], str(maxDepth[k]), str(n_estimators[l]), str(max_features[m]),
                                    str(results.sensitivity), str(results.fallout), str(results.specificity),
                                    str(results.missRate), str(results.testErr), str(results.AUC)])

            # input("Premi invio per continuare...\n")

        # N.B. Può capitare che certi test ottengano lo stesso risultato, ma solo uno viene etichettato come migliore
        print("\nMiglior risultato RandomForestModel Sklearn : test " + str(index_min_err+1) + "." + str(iter_min_err+1))
        print("Impurity: " + impurity[j_min_err] + ", maxDepth: " + str(maxDepth[k_min_err]) + ", n_estimators: "
              + str(n_estimators[l_min_err]) + ", max_features: " + max_features[m_min_err])
        print("Tasso di errore: " + str(result_min.testErr * 100) + "%")
        print('--------------------------------------------------\n')

        csvBestWriter.writerow(['RandomForest Sklearn', str(index_min_err+1) + "." + str(iter_min_err+1),
                                result_min.sensitivity, result_min.fallout, result_min.specificity, result_min.missRate,
                                result_min.testErr, result_min.AUC])

        csvWriter.writerow(['#############################'])

        j = -1
        k = -1
        l = -1
        m = -1
        index_min_err = -1
        iter_min_err = -1
        j_min_err = -1
        k_min_err = -1
        l_min_err = -1
        m_min_err = -1
        result_min = ra.Results(1, 1, 1, 1, 1, 1)
        iter_count = 0

        csvWriter.writerow(['Gradient Boosted tree'])
        csvWriter.writerow(['Iteration', 'Loss', 'Max_Depth', 'Max_Bins', 'Num_Iterations',
                            'Sensitivity', 'Fallout', 'Specificity', 'Miss_rate',
                            'Test_Error', 'AUC'])

        # loss = ['logLoss', 'leastSquaresError', 'leastAbsoluteError']
        # maxDepth2 = [3, 5]
        # maxBins = [32, 64, 128]
        # numIterations = [50, 100]

        # GRADIENT BOOSTED TREE ML.classification
        max_count = 35
        for i in range(0, 35):
            j = int(i / 12)     # loss
            k = int(i / 6) % 2  # maxDepth2
            l = int(i / 2) % 3  # maxBins
            m = i % 2           # numIterators

            for t in range(0, multiplier):
                if verbose:
                    print("\n--------------------------------------------------\n")
                    print("Test " + str(i+1) + "." + str(t+1)
                          + " con loss: " + loss[j] + ", maxDepth: " + str(maxDepth2[k])
                          + ", maxBins: " + str(maxBins[l]) + ", numIterations: " + str(numIterations[m]))
                else:
                    iter_count = (i * multiplier) + t
                    percentage = int((iter_count / (max_count * multiplier)) * 100)
                    sys.stdout.write('\rPercentuale completamento test GBT: ' + str(percentage) + "%"),
                    sys.stdout.flush()

                (trainingData, testData) = datas[t]
                c_trainingData = trainingData.map(lambda x: LabeledPoint(x[30], x[:30]))
                c_testData = trainingData.map(lambda x: LabeledPoint(x[30], x[:30]))
                testRecordsNumber = float(testData.count())

                labelsAndPredictions = cgbt.gradientBoostedTrees(c_trainingData, c_testData, loss[j], numIterations[m],
                                                                 maxDepth2[k], maxBins[l])

                results = ra.resultAnalisys(labelsAndPredictions, testRecordsNumber, verbose)

                if results.testErr < result_min.testErr:
                    index_min_err = i
                    iter_min_err = t
                    j_min_err = j
                    k_min_err = k
                    l_min_err = l
                    m_min_err = m
                    result_min = results

                csvWriter.writerow([str(i+1) + "." + str(t+1),
                                    loss[j], str(maxDepth2[k]), str(maxBins[l]), str(numIterations[m]),
                                    str(results.sensitivity), str(results.fallout), str(results.specificity),
                                    str(results.missRate), str(results.testErr), str(results.AUC)])

            # input("Premi invio per continuare...\n")

            # N.B. Può capitare che certi test ottengano lo stesso risultato, ma solo uno viene etichettato come migliore
        print("\nMiglior risultato GradientBoostedTree: test " + str(index_min_err+1) + "." + str(iter_min_err+1))
        print("Loss: " + loss[j_min_err] + ", maxDepth: " + str(maxDepth2[k_min_err]) + ", maxBins: "
              + str(maxBins[l_min_err]) + ", numIterations: " + str(numIterations[m_min_err]))
        print("Tasso di errore: " + str(result_min.testErr * 100) + "%")
        print('--------------------------------------------------\n')

        csvBestWriter.writerow(['GradientBoostedTree', str(index_min_err+1) + "." + str(iter_min_err+1),
                                result_min.sensitivity, result_min.fallout, result_min.specificity, result_min.missRate,
                                result_min.testErr, result_min.AUC])

        csvWriter.writerow(['#############################'])

        j = -1
        k = -1
        l = -1
        m = -1
        index_min_err = -1
        iter_min_err = -1
        j_min_err = -1
        k_min_err = -1
        l_min_err = -1
        m_min_err = -1
        result_min = ra.Results(1, 1, 1, 1, 1, 1)
        iter_count = 0

        csvWriter.writerow(['Multilayer Perceptron'])
        csvWriter.writerow(['Iteration', 'Max_Iter', 'Layer', 'Block_Size', 'Solver',
                            'Sensitivity', 'Fallout', 'Specificity', 'Miss_rate',
                            'Test_Error', 'AUC'])

        # maxIter (numIterations) = [50, 100]
        # layer = [[30, 10, 2], [30, 20, 2], [30, 20, 10, 2]]
        # blockSize = [128, 256, 512]
        # solver = ['l-bfgs', 'gd']


        # MULTILAYER PERCEPTRON ML.classification
        max_count = 35
        for i in range(0, 35):
            j = int(i / 18)     # numIterator
            k = int(i / 6) % 3  # layer
            l = int(i / 2) % 3  # blockSize
            m = i % 2           # solver

            for t in range(0, multiplier):
                if verbose:
                    print("\n--------------------------------------------------\n")
                    print("Test " + str(i+1) + "." + str(t+1)
                          + " con maxIter: " + str(numIterations[j]) + ", layer: " + str(layer[k])
                          + ", blockSize: " + str(blockSize[l]) + ", solver: " + solver[m])
                else:
                    iter_count = (i * multiplier) + t
                    percentage = int((iter_count / (max_count * multiplier)) * 100)
                    sys.stdout.write('\rPercentuale completamento test MLP: ' + str(percentage) + "%"),
                    sys.stdout.flush()

                (trainingData, testData) = datas[t]
                testRecordsNumber = float(testData.count())

                labelsAndPredictions = cmlp.multilayerPerceptron(trainingData, testData, numIterations[j], layer[k],
                                                                 blockSize[l], solver[m])

                # x = labelsAndPredictions.collect()

                results = ra.resultAnalisys(labelsAndPredictions, testRecordsNumber, verbose)

                if results.testErr < result_min.testErr:
                    index_min_err = i
                    iter_min_err = t
                    j_min_err = j
                    l_min_err = l
                    k_min_err = k
                    m_min_err = m
                    result_min = results

                csvWriter.writerow([str(i+1) + "." + str(t+1),
                                    str(numIterations[j]), str(layer[k]), str(blockSize[l]), solver[m],
                                    str(results.sensitivity), str(results.fallout), str(results.specificity),
                                    str(results.missRate), str(results.testErr), str(results.AUC)])

            # input("Premi invio per continuare...\n")

        # N.B. Può capitare che certi test ottengano lo stesso risultato, ma solo uno viene etichettato come migliore
        print("\nMiglior risultato Multilayer Perceptron: test " + str(index_min_err+1) + "." + str(iter_min_err+1))
        print("maxIter: " + str(numIterations[j_min_err]) + ", layer: " + str(layer[k_min_err]) + ", blockSize: "
              + str(blockSize[l_min_err]) + ", solver: " + solver[m_min_err])
        print("Tasso di errore: " + str(result_min.testErr * 100) + "%")
        print('--------------------------------------------------\n')

        csvBestWriter.writerow(['Multilayer Perceptron', str(index_min_err+1) + "." + str(iter_min_err+1),
                                result_min.sensitivity, result_min.fallout, result_min.specificity, result_min.missRate,
                                result_min.testErr, result_min.AUC])

        csvWriter.writerow(['#############################'])
