# coding=utf-8
from pyspark.ml import Pipeline
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

    validator = None
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
        validator = crossVal
    else:
        validator = lrc

    print("    -validator creato")

    training = trainingData.map(lambda x: (x[31], Vectors.dense(x[:30]), x[30])).toDF(
        schema=['index', 'features', 'label']).orderBy('index')

    # Configure an ML pipeline, which consists of three stages: tokenizer, hashingTF, and lr.
    # tokenizer = Tokenizer(inputCol="features", outputCol="transactions")
    # hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol="rawFeatures", numFeatures=29)

    pipeline = Pipeline(stages=[validator])

    model = pipeline.fit(training)

    print("    -modello addestrato con la pipeline (" + str(training.count()) + " elementi utilizzati come training)")

    test = testData.map(lambda x: (x[30], Vectors.dense(x[:30]), x[31])).toDF(
        schema=['label', 'features', 'index']).orderBy('index')

    # prediction = predictions, label, index
    predictionsAndLabels = model.transform(test).rdd.map(lambda x: (x[5], x[0], x[2]))

    print("    -" + str(predictionsAndLabels.count()) + " elementi predetti (" + str(
        test.count()) + " elementi usati come test)")

    return predictionsAndLabels