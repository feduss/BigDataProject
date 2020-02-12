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
def multilayerPerceptron(trainingData, testData, maxIter=100, layers=None, blockSize=128, seed=None,
                         featuresCol="features", labelCol="label", predictionCol="prediction", tol=1e-6, stepSize=0.03,
                         solver="l-bfgs", initialWeights=None, probabilityCol="probability",
                         rawPredictionCol="rawPrediction"):

    # specify layers for the neural network:
    # input layer of size 4 (features), two intermediate of size 5 and 4
    # and output of size 3 (classes)
    layers = [4, 5, 4, 3]

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

    # compute accuracy on the test set
    result = model.transform(testData)
    labelsAndPredictions = result.select("prediction", "label")
    x = 0

if __name__ == '__main__' :
    (trainingData, testData, c_trainingData, c_testData) = SetsCreation.setsCreation()
    multilayerPerceptron(trainingData, testData, 100, None, 128)