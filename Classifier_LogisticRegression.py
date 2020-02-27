from pyspark.ml.classification import LogisticRegression
from pyspark.ml.linalg import Vectors


# Parametri
# I Parametri con "Col" sono nomi di colonna...da ignorare
# maxIter = numero max di iterazioni
# regParam = parametro di regolarizzazione ( >=0 )
# elasticNetParam = parametro di rete elastica mista (0-1)
# tol = tolleranza di convergenza
# fitIntercept = il fit ha un termine di intercettazione (?)
# threshold = Si usa nella predizione binaria (0-1)
# thresholds = //////////////////////////
# standardization = se standardizzare le features prima di fare il fit
# aggregationDepth = profondità suggerita per l'aggregazione degli alberi ( >=2 )
# family = algoritmo da usare (auto, binomial e multinomial)
def logisticRegression(trainingData, testData, featuresCol="features", labelCol="label", predictionCol="prediction",
                       maxIter=100, regParam=0.0, elasticNetParam=0.0, tol=1e-6, fitIntercept=True,
                       threshold=0.5, thresholds=None, probabilityCol="probability", rawPredictionCol="rawPrediction",
                       standardization=False, weightCol=None, aggregationDepth=2, family="binomial"):


    testRecordsNumber = float(len(testData.collect()))

    lrm = LogisticRegression(maxIter=maxIter, regParam=regParam, elasticNetParam=elasticNetParam,
                             standardization = standardization, aggregationDepth = aggregationDepth, family = family)


    # Separo le classi (label) dalle features per il trainingSet
    trainingLabels = trainingData.map(lambda x: x[30])
    trainingFeatures = trainingData.map(lambda x: x[:30])

    # Creo un dataframe [features:vector, label: double] (Vectors.dense rimuove un livello di annidamento)
    trainingData = trainingFeatures.map(lambda x: Vectors.dense(x)).zip(trainingLabels).toDF(
        schema=['features', 'label'])

    model = lrm.fit(trainingData)

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