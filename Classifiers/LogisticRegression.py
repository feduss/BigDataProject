# coding=utf-8
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.linalg import Vectors
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator


# https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.classification.LogisticRegression
# featuresCol               : Features column name
# labelCol                  : Label column name
# predictionCol             : Prediction column name
# maxIter                   : Max number of iterations (>= 0)
# regParam                  : Regularization parameter (>= 0)
# elasticNetParam           : The ElasticNet mixing parameter, in range [0, 1].
#                             For alpha = 0, the penalty is an L2 penalty. For alpha = 1, it is an L1 penalty
# tol                       : The convergence tolerance for iterative algorithms (>= 0)
# fitIntercept              : Whether to fit an intercept term
# threshold                 : Threshold in binary classification prediction, in range [0, 1].
#                             If threshold and thresholds are both set, they must match.
#                             e.g. if threshold is p, then thresholds must be equal to [1-p, p]
# thresholds                : Thresholds in multi-class classification to adjust the probability of predicting
#                             each class. Array must have length equal to the number of classes, with values > 0,
#                             excepting that at most one value may be 0.
#                             The class with largest value p/t is predicted, where p is the original probability
#                             of that class and t is the class's threshold
# probabilityCol            : Column name for predicted class conditional probabilities.
# rawPredictionCol          : Raw prediction (a.k.a. confidence) column name
# standardization           : Whether to standardize the training features before fitting the model
# weightCol                 : Weight column name. If this is not set or empty, we treat all instance weights as 1.0
# aggregationDepth          : Suggested depth for treeAggregate (>= 2)
# family                    : The name of family which is a description of the label distribution to be used
#                             in the model. Supported options: auto, binomial, multinomial
# lowerBoundsOnCoefficients : The lower bounds on coefficients if fitting under bound constrained optimization.
#                             The bound matrix must be compatible with the shape (1, number of features) for binomial
#                             regression, or (number of classes, number of features) for multinomial regression
# upperBoundsOnCoefficients : The upper bounds on coefficients if fitting under bound constrained optimization.
#                             The bound matrix must be compatible with the shape (1, number of features) for binomial
#                             regression, or (number of classes, number of features) for multinomial regression
# lowerBoundsOnIntercepts   : The lower bounds on intercepts if fitting under bound constrained optimization.
#                             The bounds vector size must beequal with 1 for binomial regression, or the number
#                             of classes for multinomial regression
# upperBoundsOnIntercepts   : The upper bounds on intercepts if fitting under bound constrained optimization.
#                             The bound vector size must be equal with 1 for binomial regression, or the number
#                             of classes for multinomial regression

def logisticRegression(trainingData, testData, maxIter, regParam, elasticNetParam, aggregationDepth,
                       enableCrossValidator=False, featuresCol="features", labelCol="label",
                       predictionCol="prediction", tol=1e-6, fitIntercept=True, threshold=0.5,
                       probabilityCol="probability", rawPredictionCol="rawPrediction", standardization=False,
                       family="binomial"):

    print("Inizio classificazione con LogisticRegressionClassifier")

    # Inizializzo il modello del classificatore con i parametri in input (e quelli default)
    lrc = LogisticRegression(maxIter=maxIter, regParam=regParam, elasticNetParam=elasticNetParam,
                             aggregationDepth=aggregationDepth, featuresCol=featuresCol, labelCol=labelCol,
                             predictionCol=predictionCol, tol=tol, fitIntercept=fitIntercept,
                             threshold=threshold, probabilityCol=probabilityCol,
                             rawPredictionCol=rawPredictionCol, standardization=standardization, family=family)

    print("    -modello creato")

    # In caso di cross validation
    if enableCrossValidator:
        # Creo la mappa dei parametri
        paramGrid = ParamGridBuilder().build()

        # Inizializzo l'evaluator
        evaluator = BinaryClassificationEvaluator()

        # Creo il sistema di k-fold cross validation, dove estiamtor è il classificatore da valutare e numFolds è il K
        crossVal = CrossValidator(estimator=lrc,
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
        model = lrc.fit(trainingData)

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

    # result = features, label, rawPrediction, probability, prediction
    # result.rdd.map = prediction, label, features
    result = result.rdd.map(lambda x: (x[0], (x[4], x[1])))

    # indicesAndFeatures = index, features
    # indicesAndFeatures = features, index
    indicesAndFeatures = indicesAndFeatures.rdd.map(lambda x: (x[1], x[0]))

    # join = predictions, labels, index
    predictionsAndLabels = result.join(indicesAndFeatures).map(lambda x: (x[1][0][0], x[1][0][1], x[1][1]))

    print("    -LR number of predictionsAndLabels elements: " + str(predictionsAndLabels.count()))

    # index, (pred, lab)
    predictionsAndLabels = predictionsAndLabels.sortByKey(ascending=True)
    # pred, lab, index
    predictionsAndLabels = predictionsAndLabels.map(lambda x: (x[1][0], x[1][1], x[0]))

    return predictionsAndLabels

    return predictionsAndLabels