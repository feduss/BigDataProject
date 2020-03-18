from pyspark.ml.linalg import Vectors
from pyspark.shell import sc
import SetsCreation
from pyspark.mllib.tree import GradientBoostedTrees
from pyspark.ml.classification import GBTClassifier


# Parametri (per la versione mllib):
# categoricalFeaturesInfo = ? (come per il decisionTree)
# loss = tipo di funzione utilizzata durante il processo di gradient boosting (il miglioramento, perchè, a partire da
#        un albero creato, cerca di crearne un'altro migliore, raffinandolo) | DEFAULT “logLoss”
# numIterations = numero di iterazioni di miglioramento | DEFAULT 100
# learningRate = Tasso di apprendimento per definire il contributo di ciascun albero interno al classificatore
#                Valori tra (0, 1] | DEFAULT 0.1
# maxDepth = profondità dei singoli alberi | DEFAULT 3
# maxBins = numero di condizioni per lo splitting di un nodo ? (DA CAPIRE MEGLIO) | DEFAULT 32
#def gradientBoostedTrees(trainingData, testData, loss="logLoss", numIterations=100, maxDepth=3, maxBins=32,
#                         categoricalFeaturesInfo = {}, learningRate=0.1):
def gradientBoostedTrees(trainingData, testData, maxIter, maxDepth, maxBins, featuresCol='features', labelCol='label', predictionCol='prediction',
                         minInstancesPerNode=1, minInfoGain=0.0, maxMemoryInMB=256,
                         cacheNodeIds=False, checkpointInterval=10, lossType='logistic',
                         stepSize=0.1, seed=None, subsamplingRate=1.0, featureSubsetStrategy='all'):

    testRecordsNumber = float(len(testData.collect()))

    gbtc = GBTClassifier(featuresCol=featuresCol, labelCol=labelCol, predictionCol=predictionCol, maxDepth=maxDepth,
                         maxBins=maxBins, minInstancesPerNode=minInstancesPerNode, minInfoGain=minInfoGain,
                         maxMemoryInMB=maxMemoryInMB, cacheNodeIds=cacheNodeIds, checkpointInterval=checkpointInterval,
                         lossType=lossType, maxIter=maxIter, stepSize=stepSize, seed=seed,
                         subsamplingRate=subsamplingRate, featureSubsetStrategy=featureSubsetStrategy)

    # Separo le classi (label) dalle features per il trainingSet
    trainingLabels = trainingData.map(lambda x: x[30])
    trainingFeatures = trainingData.map(lambda x: x[:30])

    # Creo un dataframe [features:vector, label: double] (Vectors.dense rimuove un livello di annidamento)
    trainingData = trainingFeatures.map(lambda x: Vectors.dense(x)).zip(trainingLabels).toDF(
        schema=['features', 'label'])

    model = gbtc.fit(trainingData)

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
    # Train a GradientBoostedTrees model.
    #  Notes: (a) Empty categoricalFeaturesInfo indicates all features are continuous.
    #         (b) Use more iterations in practice.
    model = GradientBoostedTrees.trainClassifier(trainingData, categoricalFeaturesInfo, loss, numIterations,
                                                 learningRate, maxDepth, maxBins)

    predictions = model.predict(testData.map(lambda x: x.features))

    #labelsAndPredictions = testData.map(lambda lp: lp.label).zip(predictions)
    predictionsAndLabels = predictions.zip(testData.map(lambda data: data.label))

    return predictionsAndLabels
    
    '''