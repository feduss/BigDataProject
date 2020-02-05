import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.preprocessing as pp
import csv

from pyspark import mllib

# dataset = sns.load_dataset("credicard.csv")
dataset = pd.read_csv('creditcard.csv') #Apro il Dataset come Panda DataFrame

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

preprocessing_status = True

# Normalizzo i dati delle colonne delle transazioni (V1, V2, ...)
for name_class in dataset.columns:
    if str(name_class).startswith("V"):
        print ("Colonna:" + name_class, end="\r")
        x = under_sample[[name_class]].values.astype(float)
        min_max_scaler = pp.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        norm_sample[name_class] = x_scaled

        max_value = norm_sample[name_class].max()
        min_value = norm_sample[name_class].min()
        if(max_value > 1.1 or min_value < -1.1):
            print("La colonna " + name_class + " non è normalizzata")
            print("Valore max: " + str(max_value) +", valore min: " + str(min_value))
            preprocessing_status = False


if(preprocessing_status):
    print("Preprocessing eseguito correttamente")

# Apro il ds iniziale
with open("creditcard.csv") as original_dataset:
    csvReader = list(csv.reader(original_dataset))
    # Creo il nuovo ds
    with open('creditcard_undersampled.csv', 'w') as new_dataset:
        csvWriter = csv.writer(new_dataset)
        csvWriter.writerow(csvReader[0]) # scrivo l'header
        # Scrivo le nuove righe
        new_rows = norm_sample.values.tolist()
        csvWriter.writerows(new_rows)

print("Nuovo dataset creato correttamente")
