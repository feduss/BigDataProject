# coding=utf-8
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from pyspark.shell import sc

import SetsCreation


def randomForest(trainingData, testData, n_estimators, criterion, max_depth, max_features):

    # Creo il modello
    # n_estimators = numero di alberi nella foresta | DEFAULT 100
    # criterion = funzione per misurare la qualità dello split | DEFAULT gini
    # max_depth = profondità max dell'albero | DEFAULT None
    # min_samples_split = numero minimo di samples per splittare un nodo interno (?) | DEFAULT 2
    # min_samples_leaf = numero minimo di samples essere considerata come foglia (?) | DEFAULT 1
    # min_weight_fraction_leaf = la frazione di peso minimo dlla somma totale dei pesi per
    #                            essere una foglia (?) | DEFAULT 0.
    # max_features = il numero di feature da considerare quando si ricerca il miglior split | DEFAULT auto
    # max_leaf_nodes = numero massimo di nodi foglia | DEFAULT None
    # min_impurity_decrease = un nodo verrà splittato se lo split decrementerà l'impurità in maniera maggiore o uguale
    #                         a questo valore | DEFAULT 0.
    # min_impurity_split = un nodo verrà splittato se la sua impurità è sopra a questa soglia, altrimenti sarà
    #                      una foglia | DEFAULT 1e-7
    # bootstrap = true per usare i bootstrap samples, altrimenti usa l'intero ds per creare l'albero | DEFAULT True
    # oob_score = true per usare gli out-of-bag (????) samples per stimare la precisione | DEFAULT False
    # n_jobs = numero di jobs da eseguire in parallelo | DEFAUL None
    # random_state = seed (credo) per il generatore di numeri random | DEFAULT None
    # verbose = controlla la verbosita durante il fitting e il predicting | DEFAULT 0
    # warm_start = true per riusare aspetti del precedente modello trainato con valori di parametri | DEFAULT False
    # class_weight = peso associato ad ogni classe; se non dato, hanno lo stesso peso, pari ad 1 | DEFAUL None
    # ccp_alpha = Complexity parameter used for Minimal Cost-Complexity Pruning. The subtree with the largest cost
    #             complexity that is smaller than ccp_alpha will be chosen | DEFAULT 0.0, non dato
    # max_samples = If bootstrap is True, the number of samples to draw from X to train each base estimator|DEFAULT None

    #Per il momento uso i parametri di default
    model = RandomForestClassifier(n_estimators = n_estimators, criterion = criterion, max_depth = max_depth,
                                   max_features = max_features)

    training_features = trainingData.map(lambda x: x.features)
    test_features = testData.map(lambda x: x.features)
    training_classes = trainingData.map(lambda x: x.label)
    test_classes = testData.map(lambda x: x.label)

    # Traino il modello
    model.fit(training_features.collect(), training_classes.collect())

    test_pred = sc.parallelize(model.predict(test_features.collect()), test_classes.getNumPartitions()).map(lambda x: float(x))

    # Converto test_classes nel tipo di rdd corretto (lo stesso di test_pred) --> E' un workaround
    # Poichè la zip() richiede, citando : "the two RDDs have the same number of partitions and the same
    #         number of elements in each partition"
    x = sc.parallelize(test_classes.collect(), test_classes.getNumPartitions())

    # Unisco le label e le predizioni
    predictionsAndLabels = test_pred.zip(x)
    #labelsAndPredictions = x.zip(test_pred)

    return predictionsAndLabels

# if __name__== "__main__":
#    (trainingData, testData) = SetsCreation.setsCreation()
#    randomForest(trainingData, testData, n_estimators = 100, criterion = 'gini', max_depth = None, max_features = 'auto')