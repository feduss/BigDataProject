from pyspark.ml.classification import LinearSVC
from pyspark.shell import sc
from pyspark.sql import Row
from pyspark.ml.linalg import Vectors
import SetsCreation

#Ovviamente era solo una prova...fallita :D
def linearSVC(trainingData, testData, featuresCol='features', labelCol='label', predictionCol='prediction',
                 maxIter=100, regParam=0.0, tol=1e-06, rawPredictionCol='rawPrediction', fitIntercept=True,
                 standardization=True, threshold=0.0, weightCol=None, aggregationDepth=2):

    # Linear support vector machine classifier
    # rawPrediction è la confidence
    # I parametri non spiegati sono auto-esplicativi
    # regParam = regularization parameter (>= 0)
    # tol = the convergence tolerance for iterative algorithms (>= 0)
    # fitIntercept = whether to fit an intercept term
    # standardization = whether to standardize the training features before fitting the model
    # threshold = The threshold in binary classification applied to the linear model prediction. This threshold can
    # be any real number, where Inf will make all predictions 0.0 and -Inf will make all predictions 1.0
    # weightCol = weight column name. If this is not set or empty, we treat all instance weights as 1.0
    # aggregationDepth = suggested depth for treeAggregate (>= 2)
    #
    # Creo il classificatore
    '''lsvc = LinearSVC(featuresCol = featuresCol, labelCol = labelCol, predictionCol = predictionCol, maxIter = maxIter,
                     regParam = regParam, tol = tol, rawPredictionCol = rawPredictionCol, fitIntercept = fitIntercept,
                     standardization = standardization, threshold = threshold, weightCol = weightCol,
                     aggregationDepth = aggregationDepth)

    # Creo e traino il modello
    x = sc.parallelize(trainingData.collect()).toDF()
    lsvcModel = lsvc.fit(df)

    # Print the coefficients and intercept for linear SVC
    print("Coefficients: " + str(lsvcModel.coefficients))
    print("Intercept: " + str(lsvcModel.intercept))

    result = lsvcModel.transform(testData).head()

    print(result.prediction)'''

    a = 0


if __name__ == "__main__" :
    (trainingData, testData) = SetsCreation.setsCreation()
    linearSVC(trainingData, testData)