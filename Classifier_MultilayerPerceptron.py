from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.linalg import Vectors
from pyspark.shell import sc
import SetsCreation
from pyspark.ml.classification import MultilayerPerceptronClassifier

# Parametri:
# Tutti quelli che finiscono per Col sono nomi di colonna
# maxIter = numero massimo di iterazioni | DEFAULT 100
# layers = array con le dimensioni dei layers da quello di input a quello di output
# blockSize = Block size for stacking input data in matrices. Data is stacked within partitions.
#             If block size is more than remaining data in a partition then it is adjusted to the size of this data.
#             Recommended size is between 10 and 1000, default is 128.'
# seed = random seed
# tol = the convergence tolerance for iterative algorithms
# stepSize = Step size to be used for each iteration of optimization
# solver = The solver algorithm for optimization. Supported options: l-bfgs, gd
# initialWeights = he initial weights of the model
def multilayerPerceptron(trainingData, testData, maxIter=100, layers=None, blockSize=128, solver="l-bfgs", seed=None,
                         featuresCol="features", labelCol="label", predictionCol="prediction", tol=1e-6, stepSize=0.03,
                         initialWeights=None, probabilityCol="probability",
                         rawPredictionCol="rawPrediction"):

    # Creo il classificatore
    trainer = MultilayerPerceptronClassifier(maxIter=maxIter, layers=layers, blockSize=blockSize, solver=solver)

    #Separo le classi (label) dalle features per il trainingSet
    trainingLabels = trainingData.map(lambda x: x[30])
    trainingFeatures = trainingData.map(lambda x: x[:30])

    #creo un dataframe [features:vector, label: double] (Vectors.dense rimuove un livello di annidamento)
    trainingData = trainingFeatures.map(lambda x: Vectors.dense(x)).zip(trainingLabels).toDF(schema=['features','label'])

    # Traino il model
    model = trainer.fit(trainingData)

    # Separo le classi (label) dalle features per il trainingSet
    testLabels = testData.map(lambda x: x[30])
    testFeatures = testData.map(lambda x: x[:30])

    testData = testFeatures.map(lambda x: Vectors.dense(x)).zip(testLabels).toDF(schema=['features', 'label'])

    # Calcolo le predizioni
    result = model.transform(testData)

    #Converto i risultati nel formato corretto
    labelsAndPredictions = result.rdd.map(lambda x: x.label).zip(result.rdd.map(lambda x: x.prediction))

    return labelsAndPredictions