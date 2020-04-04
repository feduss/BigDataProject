# coding=utf-8
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

    print("    -crossValidation eseguita")

    # Separo le classi (label) dalle features per il trainingSet
    trainingLabels = trainingData.map(lambda x: x[30])
    trainingFeatures = trainingData.map(lambda x: x[:30])

    # Creo un dataframe [features:vector, label: double] (Vectors.dense rimuove un livello di annidamento)
    trainingData = trainingFeatures \
        .map(lambda x: Vectors.dense(x)).zip(trainingLabels) \
        .toDF(schema=['features', 'label'])

    # Genero il modello (addestrato attraverso la cross validation, o con i parametri in input)
    if enableCrossValidator:
        model = crossVal.fit(trainingData)
    else:
        model = lsvc.fit(trainingData)

    print("    -modello addestrato con " + str(trainingData.count()) + " elementi")

    # Separo le classi (label) dalle features per il testSet
    testLabels = testData.map(lambda x: x[30])
    testFeatures = testData.map(lambda x: x[:30])
    testIndices = testData.map(lambda x: x[31])

    # Creo un dataframe [features:vector, label: double] (Vectors.dense rimuove un livello di annidamento)
    testData = testFeatures \
        .map(lambda x: Vectors.dense(x)).zip(testLabels) \
        .toDF(schema=['features', 'label'])

    # Calcolo le predizioni
    result = model.transform(testData)

    print("    -il modello addestrato ha calcolato le predizioni di " + str(testData.count()) + " elementi")

    indicesAndFeatures = testIndices.zip(testFeatures.map(lambda x: Vectors.dense(x))).toDF(
        schema=['index', 'features'])

    result.show()

    indicesAndFeatures.show()

    # result = features, label, rawPrediction, prediction
    # result.rdd.map = prediction, label, features
    result = result.rdd.map(lambda x: (x[0], (x[3], x[1])))

    # indicesAndFeatures = index, features
    # indicesAndFeatures = features, index
    indicesAndFeatures = indicesAndFeatures.rdd.map(lambda x: (x[1], x[0]))

    # join = predictions, labels, index
    predictionsAndLabels = result.join(indicesAndFeatures).map(lambda x: (x[1][0][0], x[1][0][1], x[1][1]))

    print("    -LSVC number of predictionsAndLabels elements: " + str(predictionsAndLabels.count()))

    # index, (pred, lab)
    predictionsAndLabels = predictionsAndLabels.sortByKey(ascending=True)
    # pred, lab, index
    predictionsAndLabels = predictionsAndLabels.map(lambda x: (x[1][0], x[1][1], x[0]))

    return predictionsAndLabels

    return predictionsAndLabels