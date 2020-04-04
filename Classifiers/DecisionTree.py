# coding=utf-8
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.linalg import Vectors
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator


# https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.classification.DecisionTreeClassifier
# featuresCol         : Features column name
# labelCol            : Label column name
# predictionCol       : Prediction column name
# probabilityCol      : Column name for predicted class conditional probabilities
# rawPredictionCol    : Raw prediction (a.k.a. confidence) column name
# maxDepth            : Maximum depth of the tree. (>= 0)
#                       E.g., depth 0 means 1 leaf node; depth 1 means 1 internal node + 2 leaf nodes
# maxBins             : Max number of bins for discretizing continuous features.
#                       Must be >=2 and >= number of categories for any categorical feature
# minInstancesPerNode : Minimum number of instances each child must have after split.
#                       If a split causes the left or right child to have fewer than minInstancesPerNode, the split
#                       will be discarded as invalid. Should be >= 1
# minInfoGain         : Minimum information gain for a split to be considered at a tree node
# maxMemoryInMB       : Maximum memory in MB allocated to histogram aggregation.
#                       If too small, then 1 node will be split per iteration, and its aggregates may exceed this size
# cacheNodeIds        : If false, the algorithm will pass trees to executors to match instances with nodes.
#                       If true, the algorithm will cache node IDs for each instance.
#                       Caching can speed up training of deeper trees. Users can set how often should the cache
#                       be checkpointed or disable it by setting checkpointInterval
# checkpointInterval  : Set checkpoint interval (>= 1) or disable checkpoint (-1).
#                       E.g. 10 means that the cache will get checkpointed every 10 iterations.
#                       Note: this setting will be ignored if the checkpoint directory is not set in the SparkContext
# impurity            : Criterion used for information gain calculation (case-insensitive).
#                       Supported options: entropy, gini
# seed                : Random seed

def decisionTree(trainingData, testData, impurity, maxDepth, maxBins, enableCrossValidator=False,
                 featuresCol='features', labelCol='label', predictionCol='prediction', probabilityCol='probability',
                 rawPredictionCol='rawPrediction', minInstancesPerNode=1, minInfoGain=0.0, maxMemoryInMB=256,
                 cacheNodeIds=False, checkpointInterval=10, seed=None):

    # Inizializzo il modello del classificatore con i parametri in input (e quelli default)
    dtc = DecisionTreeClassifier(featuresCol=featuresCol, labelCol=labelCol, predictionCol=predictionCol,
                                 probabilityCol=probabilityCol, rawPredictionCol=rawPredictionCol,
                                 maxDepth=maxDepth, maxBins=maxBins, minInstancesPerNode=minInstancesPerNode,
                                 minInfoGain=minInfoGain, maxMemoryInMB=maxMemoryInMB, cacheNodeIds=cacheNodeIds,
                                 checkpointInterval=checkpointInterval, impurity=impurity,
                                 seed=seed)

    # In caso di cross validation
    if enableCrossValidator:
        # Creo la mappa dei parametri
        paramGrid = ParamGridBuilder().build()

        # Inizializzo l'evaluator
        evaluator = BinaryClassificationEvaluator()

        # Creo il sistema di k-fold cross validation, dove estiamtor è il classificatore da valutare e numFolds è il K
        crossVal = CrossValidator(estimator=dtc,
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

    # Genero il modello (addestrato attraverso la cross validation, o con i parametri in input)
    if enableCrossValidator:
        model = crossVal.fit(trainingData)
    else:
        model = dtc.fit(trainingData)

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

    indicesAndFeatures = testIndices.zip(testFeatures.map(lambda x: Vectors.dense(x))).toDF(schema=['index', 'features'])

    result = result.join(indicesAndFeatures, 'features').drop('features').drop('rawPrediction').drop('probability').orderBy('Index')

    predictionsAndLabels = result.rdd.map(lambda x: (x[1], x[0], x[2]))

    # Converto i risultati nel formato corretto
    #predictionsAndLabels = result.rdd.map(lambda x: x.prediction).zip(result.rdd.map(lambda x: x.label))
                           #.zip(result.rdd.map(lambda x: x.index))\
                           #.map(lambda x: (x[0][0], x[0][1], x[1]))

    return predictionsAndLabels
