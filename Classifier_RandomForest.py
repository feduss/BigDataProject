from pyspark.mllib.tree import RandomForest

# Random Forest Classifier
# numClasses = numero di classi (nel nostro caso true e false, 0 e 1 )
# categoricalFeaturesInfo = ? (come per il decisionTree)
# numTree = numero di alberi nella foresta
# featureSubsetStrategy = featureSubsetStrategy Number of features to consider for splits at each node.
#                         Supported: "auto", "all", "sqrt", "log2", "onethird". If "auto" is set, this parameter
#                         is set based on numTrees
#                         LINK:https://spark.apache.org/docs/1.4.0/api/java/org/apache/spark/mllib/tree/RandomForest.html
# impurity = Criterio usato per il calcolo dell'information gain (default gini oppure esiste entropy)
# maxDepth = profondità dei singoli alberi
# maxBins = numero di condizioni per lo splitting di un nodo ? (DA CAPIRE MEGLIO)
# seed = Random seed for bootstrapping and choosing feature subsets.... ?
# TODO cercare i valori dei parametri su google e scrivere il loro significato qui sopra
'''
impurity = ['gini', 'entropy']
maxDepth = [5, 6, 7]
maxBins = [32, 64, 128]
numTrees = []
'''
def randomForest(trainingData, testData, impurity, maxDepth, maxBins, numTrees, numClasses=2,
                 categoricalFeaturesInfo={}, featureSubsetStrategy="auto", seed=None):


    model = RandomForest.trainClassifier(data = trainingData, numClasses= numClasses,
                                         categoricalFeaturesInfo=categoricalFeaturesInfo,
                                         numTrees=numTrees, featureSubsetStrategy=featureSubsetStrategy,
                                         impurity= impurity, maxDepth=maxDepth, maxBins=maxBins, seed=seed)

    predictions = model.predict(testData.map(lambda x: x.features))
    labelsAndPredictions = testData.map(lambda x: x.label).zip(predictions)

    return labelsAndPredictions
