# coding=utf-8
import csv
import time
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn.preprocessing as pp
from pyspark.mllib.feature import StandardScaler
from pyspark.shell import spark

doUndersampled = True # False = usa il ds completo

# dataset = sns.load_dataset("credicard.csv")
dataset = pd.read_csv(str(Path(__file__).parent) + '/CSV_Sources/creditcard.csv') #Apro il Dataset come Panda DataFrame


if(doUndersampled):
    # Calcolo il numero di frodi e non frodi presenti
    no_frauds = len(dataset[dataset['Class'] == 0])
    frauds = len(dataset[dataset['Class'] == 1])

    # Calcolo gli indici dei valori indicati come frodi e non frodi
    non_fraud_indices = dataset[dataset.Class == 0].index
    fraud_indices = dataset[dataset.Class == 1].index

    # Imposto il seed della funzione random per rendere i numeri generati sempre gli stessi
    np.random.seed(9)
    # Prendo indici di non frodi randomicamente, ma ne scelgo lo stesso numero di quelli delle frodi, per bilanciamento
    random_indices = np.random.choice(non_fraud_indices, frauds, False)

    # Concateno gli indici delle frodi con i nuovi indici delle non frodi
    under_sample_indices = np.concatenate([fraud_indices, random_indices])

    # Ottengo il nuovo dataset undersampled
    under_sample = dataset.loc[under_sample_indices]

    norm_sample = under_sample
else:
    norm_sample = dataset

# Metodo 1
# Normalizzo i dati delle colonne delle transazioni (V1, V2, ...)

start_time = time.time()
norm_sample.insert(31, "Index", range(1, len(norm_sample['Time']) + 1))
for name_class in dataset.columns:
    if str(name_class).startswith("V"):
        print ("Colonna:" + name_class, end="\r")
        x = norm_sample[[name_class]].values.astype(float)
        min_max_scaler = pp.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        norm_sample[name_class] = x_scaled

        max_value = norm_sample[name_class].max()
        min_value = norm_sample[name_class].min()
        if(max_value > 1.1 or min_value < -1.1):
            print("La colonna " + name_class + " non è normalizzata")
            print("Valore max: " + str(max_value) +", valore min: " + str(min_value))

# Apro il ds iniziale
with open(str(Path(__file__).parent) + "/CSV_Sources/creditcard.csv") as original_dataset:
    csvReader = list(csv.reader(original_dataset))
    # Creo il nuovo ds
    file = "error"
    if(doUndersampled): file = "creditcard_undersampled1.csv"
    else: file = "creditcard_normalized1.csv"
    with open(str(Path(__file__).parent) + '/CSV_Sources/' + file, 'w') as new_dataset:
        csvWriter = csv.writer(new_dataset)
        csvWriter.writerow(csvReader[0] + ['Index']) # scrivo l'header
        # Scrivo le nuove righe
        new_rows = norm_sample.values.tolist()
        csvWriter.writerows(new_rows)
end_time = time.time() - start_time
print ("Tempo metodo 1: " + str(end_time) + "s")

'''
#Metodo 2
start_time = time.time()
preprocessing_status = True

scaler = StandardScaler(withMean=True, withStd=True)

# Converto il ds pandas in un ds spark
under_sample = spark.createDataFrame(under_sample).rdd.map(list)

# Estrapolo le features (tranne time, cioè la prima) e le label dalla rdd
features = under_sample.map(lambda x: x[1:30])
labels = under_sample.map(lambda x: x[30]).map(lambda x: float(x))

# Creo lo scaler con media e deviazione standard in base alle features ottenute prima
model = scaler.fit(features)

# Scalo i valori delle features
results = model.transform(features)

# Estrapolo V1, da usare come indice per la join successiva e poi time (non scalato), per motivi di ordine nel df
v1s = results.map(lambda x: x[0]).map(lambda x: float(x))
times = under_sample.map(lambda x: x[0]).map(lambda x: float(x))

# Trasformo le features scalata in un DataFrame spark
results = results.map(lambda x: x.toArray().tolist()).toDF(schema = ['V1','V2','V3','V4','V5','V6','V7','V8','V9','V10','V11','V12','V13','V14','V15','V16'
    ,'V17','V18','V19','V20','V21','V22','V23','V24','V25','V26','V27','V28','Amount'])

# Creo un DataFrame spark con time e v1
timeDF = spark.createDataFrame(zip(times.collect(), v1s.collect()), schema= ['Time', 'V1'])

# Joino i due df, per aggiungere la colonna del Time (il tutto per non normalizzarla)
norm_sample = timeDF.join(results, results.V1 == timeDF.V1, how='left').drop(timeDF.V1)

labelsDF = spark.createDataFrame(zip(times.collect(), labels.collect()), schema= ['Time', 'Class'])

# Unisco i due DataFrame spark in base al time
norm_sample = norm_sample.join(labelsDF, norm_sample.Time == labelsDF.Time, how='left').drop(labelsDF.Time).orderBy('Time', ascending = True)

# results.show()
# labelsDF.show()
# norm_sample.show()

# Apro il ds iniziale
with open(str(Path(__file__).parent) + "/CSV_Sources/creditcard.csv") as original_dataset:
    csvReader = list(csv.reader(original_dataset))
    # Creo il nuovo ds
    #with open(str(Path(__file__).parent) + '/CSV_Sources/creditcard_undersampled2.csv', 'w') as new_dataset:
    with open(str(Path(__file__).parent) + '/CSV_Sources/creditcard_normalized2.csv', 'w') as new_dataset:
        csvWriter = csv.writer(new_dataset)
        csvWriter.writerow(csvReader[0]) # scrivo l'header
        # Scrivo le nuove righe
        new_rows = norm_sample.collect()
        csvWriter.writerows(new_rows)

end_time = time.time() - start_time
print ("Tempo metodo 2: " + str(end_time) + "s")
'''