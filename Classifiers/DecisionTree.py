from pyspark.ml.linalg import Vectors
from pyspark.mllib.tree import DecisionTree
from pyspark.ml.classification import DecisionTreeClassifier

# Metodo che computa il classificatore (per la versione mllib)
# trainingData = inputdata
# numClasses = numero di classi (nel nostro caso true e false, 0 e 1 )
# categoricalFeaturesInfo = ?
# impurity = Criterio usato per il calcolo dell'information gain (default gini oppure esiste entropy)
# maxDepth = profondità dei singoli alberi
# maxBins = numero di condizioni per lo splitting di un nodo ? (DA CAPIRE MEGLIO)
# minInstancesPerNode = numero minimo di figli di un nodo parent per essere splittato
# minInfoGain = numero minimo di info ottenute per splittare un nodo


#def decisionTree(trainingData, testData, impurity, maxDepth, maxBins, minInstancesPerNode = 1, minInfoGain = 0.0,
#                 numClasses = 2, categoricalFeaturesInfo={}):

def decisionTree(trainingData, testData, impurity, maxDepth, maxBins, featuresCol='features', labelCol='label', predictionCol='prediction', probabilityCol='probability',
                 rawPredictionCol='rawPrediction', minInstancesPerNode=1, minInfoGain=0.0,
                 maxMemoryInMB=256, cacheNodeIds=False, checkpointInterval=10, seed=None):


    testRecordsNumber = float(len(testData.collect()))

    dtc =  DecisionTreeClassifier(featuresCol=featuresCol, labelCol=labelCol, predictionCol=predictionCol,
                 probabilityCol=probabilityCol, rawPredictionCol=rawPredictionCol,
                 maxDepth=maxDepth, maxBins=maxBins, minInstancesPerNode=minInstancesPerNode, minInfoGain=minInfoGain,
                 maxMemoryInMB=maxMemoryInMB, cacheNodeIds=cacheNodeIds, checkpointInterval=checkpointInterval,
                 impurity=impurity, seed=seed)

    # Separo le classi (label) dalle features per il trainingSet
    trainingLabels = trainingData.map(lambda x: x[30])
    trainingFeatures = trainingData.map(lambda x: x[:30])

    # Creo un dataframe [features:vector, label: double] (Vectors.dense rimuove un livello di annidamento)
    trainingData = trainingFeatures.map(lambda x: Vectors.dense(x)).zip(trainingLabels).toDF(
        schema=['features', 'label'])

    model = dtc.fit(trainingData)

    # Separo le classi (label) dalle features per il trainingSet
    testLabels = testData.map(lambda x: x[30])
    testFeatures = testData.map(lambda x: x[:30])

    testData = testFeatures.map(lambda x: Vectors.dense(x)).zip(testLabels).toDF(schema=['features', 'label'])

    # Calcolo le predizioni
    result = model.transform(testData)

    # Converto i risultati nel formato corretto
    # labelsAndPredictions = result.rdd.map(lambda x: x.label).zip(result.rdd.map(lambda x: x.prediction))
    predictionsAndLabels = result.rdd.map(lambda x: x.prediction).zip(result.rdd.map(lambda x: x.label))

    return predictionsAndLabels

    '''
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
    '''
