# coding=utf-8
import math
from pathlib import Path
import pandas as pd
from pyspark.shell import spark


def setsCreation(multiplier, dataset):
    if dataset == 1:
        source_file = "creditcard_undersampled" + str(dataset)
    elif dataset == 2:
        source_file = "creditcard_undersampled" + str(dataset)
    elif dataset == 3:
        source_file = "creditcard_normalized" + str(dataset - 2)
    else:
        source_file = "creditcard_normalized" + str(dataset - 2)

    datas = []

    # Leggo il ds con pandas
    p_df = pd.read_csv(str(Path(__file__).parent) + "/CSV_Sources/" + source_file + ".csv")

    # Converto il ds pandas in un ds spark, dando un numero di partition pari alla radice quadrata degli elementi
    s_df = spark.createDataFrame(p_df).repartition(int(math.sqrt(len(p_df))))

    for i in range(0, multiplier):
        datas.append(s_df.rdd.randomSplit([0.7, 0.3], seed=1234))

    # Creo una RDD di LabeledPoint
    # converted_data = s_df.rdd.map(lambda x: LabeledPoint(x[30], x[:30]))

    # Splitto i dati in training set e test set
    # (trainingData, testData) = converted_data.randomSplit([0.7, 0.3])

    return datas
