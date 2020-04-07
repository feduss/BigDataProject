# coding=utf-8
from pyspark.ml import Pipeline
from pyspark.ml.classification import LinearSVC
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.linalg import Vectors
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator


# https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.classification.LinearSVC
# featuresCol      : Features column name
# labelCol         : Label column name
# predictionCol    : Prediction column name
# maxIter          : Max number of iterations (>= 0)
# regParam         : Regularization parameter (>= 0)
# tol              : The convergence tolerance for iterative algorithms (>= 0)
# rawPredictionCol : Raw prediction (a.k.a. confidence) column name
# fitIntercept     : Whether to fit an intercept term
# standardization  : Whether to standardize the training features before fitting the model
# threshold        : The threshold in binary classification applied to the linear model prediction.
#                    This threshold can be any real number, where Inf will make all predictions 0.0
#                    and -Inf will make all predictions 1.0
# weightCol        : Weight column name. If this is not set or empty, we treat all instance weights as 1.0
# aggregationDepth : Suggested depth for treeAggregate (>= 2)

def linearSVC(trainingData, testData, maxIter, regParam, aggregationDepth, enableCrossValidator=False,
              featuresCol="features", labelCol="label", predictionCol="prediction", tol=1e-6,
              rawPredictionCol="rawPrediction", fitIntercept = True, standardization=False, threshold=0.0):

    print("Inizio classificazione con LinearSVSClassifier")

    # Inizializzo il modello del classificatore con i parametri in input (e quelli default)
    lsvc = LinearSVC(featuresCol=featuresCol, labelCol=labelCol, predictionCol=predictionCol, maxIter=maxIter,
                     regParam=regParam, tol=tol, rawPredictionCol=rawPredictionCol, fitIntercept=fitIntercept,
                     standardization=standardization, threshold=threshold, aggregationDepth=aggregationDepth)

    print("    -modello creato")

    validator = None
    # In caso di cross validation
    if enableCrossValidator:
        # Creo la mappa dei parametri
        paramGrid = ParamGridBuilder().build()

        # Inizializzo l'evaluator
        evaluator = BinaryClassificationEvaluator()

        # Creo il sistema di k-fold cross validation, dove estiamtor è il classificatore da valutare e numFolds è il K
        crossVal = CrossValidator(estimator=lsvc,
                                  estimatorParamMaps=paramGrid,
                                  evaluator=evaluator,
                                  numFolds=5)  # use 3+ folds in practice
        validator = crossVal
    else:
        validator = lsvc

    print("    -validator creato")

    training = trainingData.map(lambda x: (x[31], Vectors.dense(x[1:29]), x[30])).toDF(
        schema=['index', 'features', 'label']).orderBy('index')

    # Configure an ML pipeline, which consists of three stages: tokenizer, hashingTF, and lr.
    # tokenizer = Tokenizer(inputCol="features", outputCol="transactions")
    # hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol="rawFeatures", numFeatures=29)

    pipeline = Pipeline(stages=[validator])

    model = pipeline.fit(training)

    print("    -modello addestrato con la pipeline (" + str(training.count()) + " elementi utilizzati come training)")

    test = testData.map(lambda x: (x[30], Vectors.dense(x[1:29]), x[31])).toDF(
        schema=['label', 'features', 'index']).orderBy('index')

    # prediction = predictions, label, index
    predictionsAndLabels = model.transform(test).rdd.map(lambda x: (x[4], x[0], x[2]))

    print("    -" + str(predictionsAndLabels.count()) + " elementi predetti (" + str(
        test.count()) + " elementi usati come test)")

    return predictionsAndLabels