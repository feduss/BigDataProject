# coding=utf-8
import sys
import time
import TestClassifiers as tc
import ResultAnalysis as resa
import RunAnalysis as ra


def runTesting(used_dataset, dataset_code):
    verbose = False
    multiplier = 5
    classifiers = 5

    metric_file = "classifiers_metrics"
    analysis_file = "results"

    mainTime = time.time()
    tc.mainTestClassifier(destination_file=metric_file + dataset_code,
                          verbose=verbose, multiplier=multiplier, used_dataset=used_dataset)
    print("Test eseguiti in " + str(time.time() - mainTime) + " secondi")

    resa.ResultAnalysis(source_file=metric_file + dataset_code,
                        destination_file=analysis_file + dataset_code,
                        classifiers=classifiers)
    print("Analisi ultimate. File Results pronto")


if __name__ == "__main__":
    dataset = ""

    if len(sys.argv) <= 1:
        print("--> Specificare il tipo di dataset da utilizzare.\n"
              "--> Opzioni disponibili: 'undersampled' e 'normalized'")
        exit(1)
    elif len(sys.argv) == 2:
        dataset = sys.argv[1]
    else:
        print("--> Troppi argomenti specificati.\n"
              "--> Inserire solo il tipo di dataset da utilizzare.\n"
              "--> Opzioni disponibili: 'undersampled' e 'normalized'")
        exit(1)

    used_dataset = 0
    dataset_code = ""

    if dataset != "undersampled" and dataset != "normalized":
        print("--> Parametro erroneo.\n"
              "--> Opzioni disponibili per il dataset: 'undersampled' e 'normalized'")
        exit(1)
    elif dataset == "undersampled":
        used_dataset = 1
        dataset_code = "_u"
    else:
        used_dataset = 3
        dataset_code = "_n"

    runTesting(used_dataset, dataset_code)
    ra.runAnalysis(used_dataset, dataset_code)
