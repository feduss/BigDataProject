import pandas as pd
from pyspark.shell import sc, spark
from pyspark.mllib.regression import LabeledPoint #il decisiontree vuole una rdd di labeledpoint

def setsCreation():

    # Leggo il ds con pandas
    p_df = pd.read_csv('creditcard_undersampled.csv')

    # Converto il ds pandas in un ds spark
    s_df = spark.createDataFrame(p_df)

    (trainingData, testData) = s_df.rdd.randomSplit([0.7, 0.3])

    c_trainingData = trainingData.map(lambda x: LabeledPoint(x[30], x[:30]))
    c_testData = trainingData.map(lambda x: LabeledPoint(x[30], x[:30]))

    # Creo una RDD di LabeledPoint
    #converted_data = s_df.rdd.map(lambda x: LabeledPoint(x[30], x[:30]))

    # Splitto i dati in training set e test set
    #(trainingData, testData) = converted_data.randomSplit([0.7, 0.3])

    return (trainingData, testData, c_trainingData, c_testData)