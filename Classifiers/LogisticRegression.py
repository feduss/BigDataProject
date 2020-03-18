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

def logisticRegression(trainingData, testData, featuresCol="features", labelCol="label", predictionCol="prediction",
                       maxIter=100, regParam=0.0, elasticNetParam=0.0, tol=1e-6, fitIntercept=True,
                       threshold=0.5, thresholds=None, probabilityCol="probability", rawPredictionCol="rawPrediction",
                       standardization=False, weightCol=None, aggregationDepth=2, family="binomial",
                       lowerBoundsOnCoefficients=None, upperBoundsOnCoefficients=None, lowerBoundsOnIntercepts=None,
                       upperBoundsOnIntercepts=None):

    # Inizializzo il modello del classificatore con i parametri in input (e quelli default)
    lrc = LogisticRegression(featuresCol=featuresCol, labelCol=labelCol, predictionCol=predictionCol, maxIter=maxIter,
                             regParam=regParam, elasticNetParam=elasticNetParam, tol=tol, fitIntercept=fitIntercept,
                             threshold=threshold, thresholds=thresholds, probabilityCol=probabilityCol,
                             rawPredictionCol=rawPredictionCol, standardization=standardization, weightCol=weightCol,
                             aggregationDepth=aggregationDepth, family=family,
                             lowerBoundsOnCoefficients=lowerBoundsOnCoefficients,
                             upperBoundsOnCoefficients=upperBoundsOnCoefficients,
                             lowerBoundsOnIntercepts=lowerBoundsOnIntercepts,
                             upperBoundsOnIntercepts=upperBoundsOnIntercepts)

    # Creo la mappa dei parametri
    # TODO RIVEDERE
    paramGrid = ParamGridBuilder().build()

    # Inizializzo l'evaluator
    evaluator = BinaryClassificationEvaluator()

    # Creo il sistema di k-fold cross validation, dove estiamtor è il classificatore da valutare e numFolds è il K
    crossVal = CrossValidator(estimator=lrc,
                              estimatorParamMaps=paramGrid,
                              evaluator=evaluator,
                              numFolds=5)  # use 3+ folds in practice

    # Separo le classi (label) dalle features per il trainingSet
    trainingLabels = trainingData.map(lambda x: x[30])
    trainingFeatures = trainingData.map(lambda x: x[:30])

    # Creo un dataframe [features:vector, label: double] (Vectors.dense rimuove un livello di annidamento)
    trainingData = trainingFeatures \
        .map(lambda x: Vectors.dense(x)).zip(trainingLabels) \
        .toDF(schema=['features', 'label'])

    # Genero il modello addestrato attraverso la cross validation
    cvModel = crossVal.fit(trainingData)
    # model = lrc.fit(trainingData)

    # Separo le classi (label) dalle features per il testSet
    testLabels = testData.map(lambda x: x[30])
    testFeatures = testData.map(lambda x: x[:30])

    # Creo un dataframe [features:vector, label: double] (Vectors.dense rimuove un livello di annidamento)
    testData = testFeatures \
        .map(lambda x: Vectors.dense(x)).zip(testLabels) \
        .toDF(schema=['features', 'label'])

    # Calcolo le predizioni
    result = cvModel.transform(testData)
    # result = model.transform(testData)

    # Converto i risultati nel formato corretto
    # labelsAndPredictions = result.rdd.map(lambda x: x.label).zip(result.rdd.map(lambda x: x.prediction))
    predictionsAndLabels = result.rdd.map(lambda x: x.prediction).zip(result.rdd.map(lambda x: x.label))

    return predictionsAndLabels
