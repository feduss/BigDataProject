from pyspark.shell import sc
import SetsCreation
from pyspark.mllib.tree import GradientBoostedTrees


# Parametri:
# categoricalFeaturesInfo = ? (come per il decisionTree)
# loss = tipo di funzione utilizzata durante il processo di gradient boosting (il miglioramento, perchè, a partire da
#        un albero creato, cerca di crearne un'altro migliore, raffinandolo) | DEFAULT “logLoss”
# numIterations = numero di iterazioni di miglioramento | DEFAULT 100
# learningRate = Tasso di apprendimento per definire il contributo di ciascun albero interno al classificatore
#                Valori tra (0, 1] | DEFAULT 0.1
# maxDepth = profondità dei singoli alberi | DEFAULT 3
# maxBins = numero di condizioni per lo splitting di un nodo ? (DA CAPIRE MEGLIO) | DEFAULT 32
def gradientBoostedTrees(trainingData, testData, loss="logLoss", numIterations=100, maxDepth=3, maxBins=32,
                         categoricalFeaturesInfo = {}, learningRate=0.1):

    # Train a GradientBoostedTrees model.
    #  Notes: (a) Empty categoricalFeaturesInfo indicates all features are continuous.
    #         (b) Use more iterations in practice.
    model = GradientBoostedTrees.trainClassifier(trainingData, categoricalFeaturesInfo, loss, numIterations,
                                                 learningRate, maxDepth, maxBins)

    predictions = model.predict(testData.map(lambda x: x.features))

    #labelsAndPredictions = testData.map(lambda lp: lp.label).zip(predictions)
    predictionsAndLabels = predictions.zip(testData.map(lambda data: data.label))

    return predictionsAndLabels