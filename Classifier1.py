from pyspark.mllib.tree import DecisionTree
import SetsCreation

(trainingData, testData) = SetsCreation.setsCreation()

#Creo e addestro il DecisionTree model
#trainingData = inputdata
#numClasses = numero di classi (nel nostro caso true e false, 0 e 1 )
#categoricalFeaturesInfo = ?
#impurity = Criterio usato per il calcolo dell'information gain (default gini oppure esiste entropy)
#maxDepth = profondità dell'albero
#maxBins = numero di condizioni per lo splitting di un nodo ? (DA CAPIRE MEGLIO)
#minInstancesPerNode = numero minimo di figli di un nodo parent per essere splittato
#minInfoGain = numero minimo di info ottenute per splittare un nodo
model = DecisionTree.trainClassifier(trainingData, numClasses=2, categoricalFeaturesInfo={},
                                     impurity='gini', maxDepth=5, maxBins=32,
                                     minInstancesPerNode = 1, minInfoGain = 0.0)

#Eseguo le predizioni sui test data, prendendo solo le feature (selezionate con la map)
#Considera che ogni riga ha (classe, [features])
predictions = model.predict(testData.map(lambda x: x.features))
#Unisco le label e le predizioni
labelsAndPredictions = testData.map(lambda lp: lp.label).zip(predictions)
#Calcolo l'errore dividendo il numero di predizioni sbagliate per il numero di dati
testErr = labelsAndPredictions.filter(lambda v_p: v_p[0] != v_p[1]).count() / float(testData.count())

#Stampe varie
print('Test Error = ' + str(testErr*100) + "%")
print('Learned classification tree model:')
print(model.toDebugString())