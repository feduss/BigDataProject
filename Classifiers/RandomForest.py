from pyspark.mllib.tree import RandomForest

from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.linalg import Vectors

# Random Forest Classifier (per la versione mllib)
# numClasses = numero di classi (nel nostro caso true e false, 0 e 1 )
# categoricalFeaturesInfo = ? (come per il decisionTree)
# numTree = numero di alberi nella foresta
# featureSubsetStrategy = featureSubsetStrategy Number of features to consider for splits at each node.
#                         Supported: "auto", "all", "sqrt", "log2", "onethird". If "auto" is set, this parameter
#                         is set based on numTrees
#                         LINK:https://spark.apache.org/docs/1.4.0/api/java/org/apache/spark/mllib/tree/RandomForest.html
# impurity = Criterio usato per il calcolo dell'information gain (default gini oppure esiste entropy)
# maxDepth = profondità dei singoli alberi
# maxBins = numero di condizioni per lo splitting di un nodo ? (DA CAPIRE MEGLIO)
# seed = Random seed for bootstrapping and choosing feature subsets.... ?


# RandomForestClassifier(featuresCol='features', labelCol='label', predictionCol='prediction',
# probabilityCol='probability', rawPredictionCol='rawPrediction', maxDepth=5, maxBins=32, minInstancesPerNode=1,
# minInfoGain=0.0, maxMemoryInMB=256, cacheNodeIds=False, checkpointInterval=10, impurity='gini', numTrees=20,
# featureSubsetStrategy='auto', seed=None, subsamplingRate=1.0)

def randomForest(trainingData, testData, impurity, maxDepth, maxBins, numTrees, featuresCol='features',
                 labelCol='label', predictionCol='prediction', probabilityCol='probability',
                 rawPredictionCol='rawPrediction', minInstancesPerNode=1, minInfoGain=0.0, maxMemoryInMB=256,
                 cacheNodeIds=False, checkpointInterval=10, featureSubsetStrategy='auto', seed=None,
                 subsamplingRate=1.0):

    rfm = RandomForestClassifier(featuresCol=featuresCol, labelCol=labelCol, predictionCol=predictionCol,
                                 probabilityCol=probabilityCol, rawPredictionCol=rawPredictionCol, maxDepth=maxDepth,
                                 maxBins=maxBins, minInstancesPerNode=minInstancesPerNode, minInfoGain=minInfoGain,
                                 maxMemoryInMB=maxMemoryInMB, cacheNodeIds=cacheNodeIds,
                                 checkpointInterval=checkpointInterval, impurity=impurity, numTrees=numTrees,
                                 featureSubsetStrategy=featureSubsetStrategy, seed=seed,
                                 subsamplingRate=subsamplingRate)

    # Separo le classi (label) dalle features per il trainingSet
    trainingLabels = trainingData.map(lambda x: x[30])
    trainingFeatures = trainingData.map(lambda x: x[:30])

    # Creo un dataframe [features:vector, label: double] (Vectors.dense rimuove un livello di annidamento)
    trainingData = trainingFeatures.map(lambda x: Vectors.dense(x)).zip(trainingLabels).toDF(
        schema=['features', 'label'])

    model = rfm.fit(trainingData)

    # Separo le classi (label) dalle features per il trainingSet
    testLabels = testData.map(lambda x: x[30])
    testFeatures = testData.map(lambda x: x[:30])

    testData = testFeatures.map(lambda x: Vectors.dense(x)).zip(testLabels).toDF(schema=['features', 'label'])

    # Calcolo le predizioni
    result = model.transform(testData)

    # model = RandomForest.trainClassifier(data = trainingData, numClasses= numClasses,
    #                                     categoricalFeaturesInfo=categoricalFeaturesInfo,
    #                                     numTrees=numTrees, featureSubsetStrategy=featureSubsetStrategy,
    #                                     impurity= impurity, maxDepth=maxDepth, maxBins=maxBins, seed=seed)

    # Converto i risultati nel formato corretto
    # labelsAndPredictions = result.rdd.map(lambda x: x.label).zip(result.rdd.map(lambda x: x.prediction))
    predictionsAndLabels = result.rdd.map(lambda x: x.prediction).zip(result.rdd.map(lambda x: x.label))

    #predictions = model.predict(testData.map(lambda x: x.features))

    #predictionsAndLabels = predictions.zip(testData.map(lambda data: data.label))
    #labelsAndPredictions = testData.map(lambda x: x.label).zip(predictions)

    return predictionsAndLabels
