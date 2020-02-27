from pyspark.ml.classification import LinearSVC
from pyspark.ml.linalg import Vectors


def linearSVC(trainingData, testData, featuresCol="features", labelCol="label", predictionCol="prediction",
                 maxIter=100, regParam=0.0, tol=1e-6, rawPredictionCol="rawPrediction",
                 fitIntercept = True, standardization = False, threshold=0.0, weightCol=None,
                 aggregationDepth=2):

    lsvc = LinearSVC(maxIter=10, regParam=0.1, standardization = standardization, aggregationDepth = aggregationDepth)

    # Separo le classi (label) dalle features per il trainingSet
    trainingLabels = trainingData.map(lambda x: x[30])
    trainingFeatures = trainingData.map(lambda x: x[:30])

    # Creo un dataframe [features:vector, label: double] (Vectors.dense rimuove un livello di annidamento)
    trainingData = trainingFeatures.map(lambda x: Vectors.dense(x)).zip(trainingLabels).toDF(
        schema=['features', 'label'])

    model = lsvc.fit(trainingData)

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