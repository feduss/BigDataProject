# coding=utf-8
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.linalg import Vectors
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator


# https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.classification.RandomForestClassifier
# featuresCol           : Features column name
# labelCol              : Label column name
# predictionCol         : Prediction column name
# probabilityCol        : Column name for predicted class conditional probabilities
# rawPredictionCol      : Raw prediction (a.k.a. confidence) column name
# maxDepth              : Maximum depth of the tree. (>= 0)
#                         E.g., depth 0 means 1 leaf node; depth 1 means 1 internal node + 2 leaf nodes
# maxBins               : Max number of bins for discretizing continuous features.
#                         Must be >=2 and >= number of categories for any categorical feature
# minInstancesPerNode   : Minimum number of instances each child must have after split.
#                         If a split causes the left or right child to have fewer than minInstancesPerNode, the split
#                         will be discarded as invalid. Should be >= 1
# minInfoGain           : Minimum information gain for a split to be considered at a tree node
# maxMemoryInMB         : Maximum memory in MB allocated to histogram aggregation.
#                         If too small, then 1 node will be split per iteration, and its aggregates may exceed this size
# cacheNodeIds          : If false, the algorithm will pass trees to executors to match instances with nodes.
#                         If true, the algorithm will cache node IDs for each instance.
#                         Caching can speed up training of deeper trees. Users can set how often should the cache
#                         be checkpointed or disable it by setting checkpointInterval
# checkpointInterval    : Set checkpoint interval (>= 1) or disable checkpoint (-1).
#                         E.g. 10 means that the cache will get checkpointed every 10 iterations.
#                         Note: this setting will be ignored if the checkpoint directory is not set in the SparkContext
# impurity              : Criterion used for information gain calculation (case-insensitive).
#                         Supported options: entropy, gini
# numTrees              : Number of trees to train (>= 1)
# featureSubsetStrategy : The number of features to consider for splits at each tree node.
#                         Supported options: 'auto' (choose automatically for task: If numTrees == 1, set to 'all'.
#                         If numTrees > 1 (forest), set to 'sqrt' for classification and to 'onethird' for regression),
#                         'all' (use all features), 'onethird' (use 1/3 of the features),
#                         'sqrt' (use sqrt(number of features)), 'log2' (use log2(number of features)),
#                         'n' (when n is in the range (0, 1.0], use n * number of features.
#                         When n is in the range (1, number of features), use n features). default = 'auto
# seed                  : Random seed
# subsamplingRate       : Fraction of the training data used for learning each decision tree, in range (0, 1]

def randomForest(trainingData, testData, impurity, maxDepth, maxBins, numTrees, enableCrossValidator=False,
                 featuresCol='features', labelCol='label', predictionCol='prediction', probabilityCol='probability',
                 rawPredictionCol='rawPrediction', minInstancesPerNode=1, minInfoGain=0.0, maxMemoryInMB=256,
                 cacheNodeIds=False, checkpointInterval=10, featureSubsetStrategy='auto', seed=None,
                 subsamplingRate=1.0):

    print("\nInizio classificazione con RandomForestClassifier")

    # Inizializzo il modello del classificatore con i parametri in input (e quelli default)
    rfc = RandomForestClassifier(featuresCol=featuresCol, labelCol=labelCol, predictionCol=predictionCol,
                                 probabilityCol=probabilityCol, rawPredictionCol=rawPredictionCol, maxDepth=maxDepth,
                                 maxBins=maxBins, minInstancesPerNode=minInstancesPerNode, minInfoGain=minInfoGain,
                                 maxMemoryInMB=maxMemoryInMB, cacheNodeIds=cacheNodeIds,
                                 checkpointInterval=checkpointInterval, impurity=impurity, numTrees=numTrees,
                                 featureSubsetStrategy=featureSubsetStrategy, seed=seed,
                                 subsamplingRate=subsamplingRate)

    print("    -modello creato")

    validator = None
    # In caso di cross validation
    if enableCrossValidator:
        # Creo la mappa dei parametri
        paramGrid = ParamGridBuilder().build()

        # Inizializzo l'evaluator
        evaluator = BinaryClassificationEvaluator()

        # Creo il sistema di k-fold cross validation, dove estiamtor è il classificatore da valutare e numFolds è il K
        crossVal = CrossValidator(estimator=rfc,
                                  estimatorParamMaps=paramGrid,
                                  evaluator=evaluator,
                                  numFolds=5)  # use 3+ folds in practice
        validator = crossVal
    else:
        validator = rfc

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
    predictionsAndLabels = model.transform(test).rdd.map(lambda x: (x[5], x[0], x[2]))

    print("    -" + str(predictionsAndLabels.count()) + " elementi predetti (" + str(
        test.count()) + " elementi usati come test)")

    return predictionsAndLabels