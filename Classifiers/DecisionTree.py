from pyspark.mllib.tree import DecisionTree

# Metodo che computa il classificatore
# trainingData = inputdata
# numClasses = numero di classi (nel nostro caso true e false, 0 e 1 )
# categoricalFeaturesInfo = ?
# impurity = Criterio usato per il calcolo dell'information gain (default gini oppure esiste entropy)
# maxDepth = profondità dei singoli alberi
# maxBins = numero di condizioni per lo splitting di un nodo ? (DA CAPIRE MEGLIO)
# minInstancesPerNode = numero minimo di figli di un nodo parent per essere splittato
# minInfoGain = numero minimo di info ottenute per splittare un nodo


def decisionTree(trainingData, testData, impurity, maxDepth, maxBins, minInstancesPerNode = 1, minInfoGain = 0.0,
                 numClasses = 2, categoricalFeaturesInfo={}):

    # Creo e addestro il DecisionTree model
    model = DecisionTree.trainClassifier(trainingData, numClasses, categoricalFeaturesInfo, impurity, maxDepth, maxBins,
                                         minInstancesPerNode, minInfoGain)

    # Eseguo le predizioni sui test data, prendendo solo le feature (selezionate con la map)
    # Considera che ogni riga ha (classe, [features])
    predictions = model.predict(testData.map(lambda x: x.features))

    # Unisco le label e le predizioni
    # labelsAndPredictions = testData.map(lambda lp: lp.label).zip(predictions)
    predictionsAndLabels = predictions.zip(testData.map(lambda data: data.label))

    # Richiamo la funzione per il calcolo dei risultati
    return predictionsAndLabels