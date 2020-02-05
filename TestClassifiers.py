import csv
import SetsCreation
import Classifier_DecisionTree as cdt
import ResultAnalysis as ra

# File per testare diversi trainingset e testset sui classificatori implementati, fornendo diversi parametri

# trainingData = inputdata
# numClasses = numero di classi (nel nostro caso true e false, 0 e 1 )
# categoricalFeaturesInfo = ?
# impurity = Criterio usato per il calcolo dell'information gain (default gini oppure esiste entropy)
# maxDepth = profondità dell'albero
# maxBins = numero di condizioni per lo splitting di un nodo ? (DA CAPIRE MEGLIO)
# minInstancesPerNode = numero minimo di figli di un nodo parent per essere splittato
# minInfoGain = numero minimo di info ottenute per splittare un nodo

(trainingData, testData) = SetsCreation.setsCreation()
testRecordsNumber = float(testData.count())

impurity = ['gini', 'entropy']
maxDepth = [5, 6, 7]
maxBins = [32, 64, 128]
minInstancesPerNode = [1]
minInfoGain = [0.0]

j = -1
k = -1
l = -1

index_min_err = -1
j_min_err = -1
k_min_err = -1
l_min_err = -1
testErrMin = 100

with open('classifiers_results.csv', 'w') as result_file:
    csvWriter = csv.writer(result_file)
    csvWriter.writerow(['Decision_Tree'])
    csvWriter.writerow(['Impurity', 'Max_Depth', 'Max_Bins',
                        'Sensitivity', 'Fallout', 'Specificity', 'Miss_rate',
                        'Test_Error', 'AUC'])

    for i in range(0, 18):
        j = 0 if i < 9 else 1
        k = int(i/3) % 3
        l = i % 3

        print("\n--------------------------------------------------\n")
        print("Test " + str(i+1) + " con impurity: " + impurity[j] + ", maxDepth: " + str(maxDepth[k]) + ", maxBins:"
              + str(maxBins[l]) + ", minInstancesPerNode: 1, minInfoGain: 1 ")

        labelsAndPredictions = cdt.decisionTree(trainingData, testData, impurity[j], maxDepth[k], maxBins[l],
                                                minInstancesPerNode[0], minInfoGain[0])

        results = ra.resultAnalisys(labelsAndPredictions, testRecordsNumber)

        if results.testErr < testErrMin:
            index_min_err = i
            j_min_err = j
            k_min_err = k
            l_min_err = l
            testErrMin = results.testErr

        csvWriter.writerow([impurity[j], str(maxDepth[k]), str(maxBins[l]),
                            str(results.sensitivity), str(results.fallout), str(results.specificity),
                            str(results.missRate), str(results.testErr), str(results.AUC)])

        # input("Premi invio per continuare...\n")

    csvWriter.writerow()
    csvWriter.writerow(['Random_Forest'])

print("\n--------------------------------------------------\n")
print("Miglior risultato: test " + str(index_min_err+1))
print("Impurity: " + impurity[j_min_err] + ", maxDepth: " + str(maxDepth[k_min_err]) + ", maxBins:"
          + str(maxBins[l_min_err]) + ", minInstancesPerNode: 1, minInfoGain: 1 ")
print("Tasso di errore: " + str(testErrMin*100) + "%")
