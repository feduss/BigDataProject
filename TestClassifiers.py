import csv
import sys
import time

import SetsCreation
from pyspark.mllib.regression import LabeledPoint
import Classifier_DecisionTree as cdt
import Classifier_RandomForest_sklearn as crf_sk
import Classifier_RandomForest as crf
import Classifier_GradientBoostedTree as cgbt
import Classifier_MultilayerPerceptron as cmlp
import MetricsEvalutation as me

verbose = False   # Per stampare o meno i risultati di tutti i test
multiplier = 3   # Ripetizioni del singolo test
used_dataset = 1  # Dataset utilizzato per creare e testare i classificatori; valori: [1, 2]

# File per testare diversi trainingset e testset sui classificatori implementati, fornendo diversi parametri
datas = SetsCreation.setsCreation(multiplier, used_dataset)

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
result_min = me.Results(1, 1, 1, 1, 1, 1)

with open('classifiers_metrics' + str(used_dataset) + '.csv', 'w') as metric_file:
    with open('best_classifiers_metrics' + str(used_dataset) + '.csv', 'w') as best_metric_file:
        csvWriter = csv.writer(metric_file)
        csvBestWriter = csv.writer(best_metric_file)

        csvBestWriter.writerow(['Model', 'Index best test', 'Sensitivity', 'Fallout',
                                'Specificity', 'Miss_Rate', 'Test_Err', 'AUC'])

        csvWriter.writerow(['Multiplier: ' + str(multiplier)])
        '''
        # DECISION TREE MLLib
        # impurity = ['gini', 'entropy']
        # maxDepth = [5, 6, 7]
        # maxBins = [32, 64, 128]

        max_count = len(impurity) * len(maxDepth) * len(maxBins)
        csvWriter.writerow(['Decision_Tree MLLib: ' + str(max_count) + ' different tests'])
        csvWriter.writerow(['Iteration', 'Impurity', 'Max_Depth', 'Max_Bins',
                            'Sensitivity', 'Fallout', 'Specificity', 'Miss_rate',
                            'Test_Error', 'AUC', 'Exec_Time'])

        for i in range(0, max_count):
            j = int(i / (len(maxDepth) * len(maxBins)))  # impurity
            k = int(i / len(maxBins)) % len(maxDepth)    # maxDepth
            l = i % len(maxBins)                         # maxBins

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
                c_testData = testData.map(lambda x: LabeledPoint(x[30], x[:30]))
                testRecordsNumber = float(testData.count())

                start_time = time.time()
                predictionsAndLabels = cdt.decisionTree(c_trainingData, c_testData, impurity[j], maxDepth[k],
                                                        maxBins[l], minInstancesPerNode[0], minInfoGain[0])
                end_time = float("{0:.3f}".format(time.time() - start_time))

                results = me.metricsEvalutation(predictionsAndLabels, testRecordsNumber, verbose)

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
                                    str(results.missRate), str(results.testErr), str(results.AUC), str(end_time)])

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
        result_min = me.Results(1, 1, 1, 1, 1, 1)
        iter_count = 0

        # RANDOM FOREST MLLib
        # impurity = ['gini', 'entropy']
        # maxDepth = [5, 6, 7]
        # maxBins = [32, 64, 128]
        # numTrees = [100, 200]

        max_count = len(impurity) * len(maxDepth) * len(maxBins) * len(numTrees)
        csvWriter.writerow(['Random_Forest MLLib: ' + str(max_count) + ' different tests'])
        csvWriter.writerow(['Iteration', 'Impurity', 'Max_Depth', 'Max_Bins', 'Num_Trees',
                            'Sensitivity', 'Fallout', 'Specificity', 'Miss_rate',
                            'Test_Error', 'AUC', 'Exec_Time'])

        for i in range(0, max_count):
            j = int(i / (len(maxDepth) * len(maxBins) * len(numTrees)))  # impurity
            k = int(i / (len(maxBins) * len(numTrees))) % len(maxDepth)  # maxDepth
            l = int(i / len(numTrees)) % len(maxBins)                    # maxBins
            m = i % len(numTrees)                                        # numTrees

            for t in range(0, multiplier):
                if verbose:
                    print("\n--------------------------------------------------\n")
                    print("Test " + str(i+1) + "." + str(t+1)
                          + " con impurity: " + impurity[j] + ", maxDepth: " + str(maxDepth[k])
                          + ", maxBins: " + str(maxBins[l]) + ", numTrees: " + str(numTrees[m]))
                else:
                    iter_count = (i * multiplier) + t
                    percentage = int((iter_count / (max_count * multiplier)) * 100)
                    sys.stdout.write('\rPercentuale completamento test RFML: ' + str(percentage) + "%"),
                    sys.stdout.flush()

                (trainingData, testData) = datas[t]
                c_trainingData = trainingData.map(lambda x: LabeledPoint(x[30], x[:30]))
                c_testData = testData.map(lambda x: LabeledPoint(x[30], x[:30]))
                testRecordsNumber = float(testData.count())

                start_time = time.time()
                predictionsAndLabels = crf.randomForest(c_trainingData, c_testData, impurity[j], maxDepth[k],
                                                        maxBins[l], numTrees[m])
                end_time = float("{0:.3f}".format(time.time() - start_time))

                results = me.metricsEvalutation(predictionsAndLabels, testRecordsNumber, verbose)

                if results.testErr < result_min.testErr:
                    index_min_err = i
                    iter_min_err = t
                    j_min_err = j
                    k_min_err = k
                    l_min_err = l
                    m_min_err = m
                    result_min = results

                csvWriter.writerow([str(i+1) + "." + str(t+1),
                                    impurity[j], str(maxDepth[k]), str(maxBins[l]), str(numTrees[m]),
                                    str(results.sensitivity), str(results.fallout), str(results.specificity),
                                    str(results.missRate), str(results.testErr), str(results.AUC), str(end_time)])

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
        result_min = me.Results(1, 1, 1, 1, 1, 1)
        iter_count = 0

        # RANDOM FOREST Sklearn
        # impurity = ['gini', 'entropy']
        # maxDepth = [5, 6, 7]
        # n_estimators = [100, 200]
        # max_features = ['auto', 'sqrt', 'log2']

        max_count = len(impurity) * len(maxDepth) * len(n_estimators) * len(max_features)
        csvWriter.writerow(['Random_Forest Sklearn: ' + str(max_count) + ' different tests'])
        csvWriter.writerow(['Iteration', 'Impurity', 'Max_Depth', 'N_Estimators', 'Max_Features',
                            'Sensitivity', 'Fallout', 'Specificity', 'Miss_rate',
                            'Test_Error', 'AUC', 'Exec_Time'])

        for i in range(0, max_count):
            j = int(i / (len(maxDepth) * len(n_estimators) * len(max_features)))  # impurity
            k = int(i / (len(n_estimators) * len(max_features))) % len(maxDepth)  # maxDepth
            l = int(i / len(max_features)) % len(n_estimators)                    # n_estimators
            m = i % len(max_features)                                             # max_features

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
                c_testData = testData.map(lambda x: LabeledPoint(x[30], x[:30]))
                testRecordsNumber = float(testData.count())

                start_time = time.time()
                predictionsAndLabels = crf_sk.randomForest(c_trainingData, c_testData, n_estimators[l], impurity[j],
                                                           maxDepth[k], max_features[m])
                end_time = float("{0:.3f}".format(time.time() - start_time))

                results = me.metricsEvalutation(predictionsAndLabels, testRecordsNumber, verbose)

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
                                    str(results.missRate), str(results.testErr), str(results.AUC), str(end_time)])

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
        result_min = me.Results(1, 1, 1, 1, 1, 1)
        iter_count = 0

        # GRADIENT BOOSTED TREE ML.classification
        # loss = ['logLoss', 'leastSquaresError', 'leastAbsoluteError']
        # maxDepth2 = [3, 5]
        # maxBins = [32, 64, 128]
        # numIterations = [50, 100]

        max_count = len(loss) * len(maxDepth2) * len(maxBins) * len(numIterations)
        csvWriter.writerow(['Gradient Boosted Tree: ' + str(max_count) + ' different tests'])
        csvWriter.writerow(['Iteration', 'Loss', 'Max_Depth', 'Max_Bins', 'Num_Iterations',
                            'Sensitivity', 'Fallout', 'Specificity', 'Miss_rate',
                            'Test_Error', 'AUC', 'Exec_Time'])

        for i in range(0, max_count):
            j = int(i / (len(maxDepth2) * len(maxBins) * len(numIterations)))  # loss
            k = int(i / (len(maxBins) * len(numIterations))) % len(maxDepth2)  # maxDepth2
            l = int(i / len(numIterations)) % len(maxBins)                     # maxBins
            m = i % len(numIterations)                                         # numIterations

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
                c_testData = testData.map(lambda x: LabeledPoint(x[30], x[:30]))
                testRecordsNumber = float(testData.count())

                start_time = time.time()
                predictionsAndLabels = cgbt.gradientBoostedTrees(c_trainingData, c_testData, loss[j], numIterations[m],
                                                                 maxDepth2[k], maxBins[l])
                end_time = float("{0:.3f}".format(time.time() - start_time))

                results = me.metricsEvalutation(predictionsAndLabels, testRecordsNumber, verbose)

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
                                    str(results.missRate), str(results.testErr), str(results.AUC), str(end_time)])

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
        '''
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
        result_min = me.Results(1, 1, 1, 1, 1, 1)
        iter_count = 0

        # MULTILAYER PERCEPTRON ML.classification
        # maxIter (numIterations) = [50, 100]
        # layer = [[30, 10, 2], [30, 20, 2], [30, 20, 10, 2]]
        # blockSize = [128, 256, 512]
        # solver = ['l-bfgs', 'gd']

        max_count = len(numIterations) * len(layer) * len(blockSize) * len(solver)
        csvWriter.writerow(['Multilayer Perceptron: ' + str(max_count) + ' different tests'])
        csvWriter.writerow(['Iteration', 'Max_Iter', 'Layer', 'Block_Size', 'Solver',
                            'Sensitivity', 'Fallout', 'Specificity', 'Miss_rate',
                            'Test_Error', 'AUC', 'Exec_Time'])

        for i in range(0, max_count):
            j = int(i / (len(layer) * len(blockSize) * len(solver)))  # numIterations
            k = int(i / (len(blockSize) * len(solver))) % len(layer)  # layer
            l = int(i / len(solver)) % len(blockSize)                 # blockSize
            m = i % len(solver)                                       # solver

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

                start_time = time.time()
                predictionsAndLabels = cmlp.multilayerPerceptron(trainingData, testData, numIterations[j], layer[k],
                                                                 blockSize[l], solver[m])
                end_time = float("{0:.3f}".format(time.time() - start_time))

                # x = labelsAndPredictions.collect()

                results = me.metricsEvalutation(predictionsAndLabels, testRecordsNumber, verbose)

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
                                    str(results.missRate), str(results.testErr), str(results.AUC), str(end_time)])

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
