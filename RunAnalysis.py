# coding=utf-8
from pyspark.shell import sc
import MajorityVote as mv
import time
import sys


def runAnalysis(used_dataset, dataset_code):
    analysis_file = "results"
    ensemble_file = "ensembles_metrics"

    num_instaces = sc._jsc.sc().getExecutorMemoryStatus().size()
    print("Instances online: " + str(num_instaces))

    start = time.time()

    print("Esecuzione getBestResults")
    bestResults = mv.getBestResults(source_file=analysis_file + dataset_code,
                                    num_classifiers=5)
    print("BestResults ottenuti")

    print("Esecuzione getLabelsAndPredictions")
    predALab = mv.getLabelsAndPredictions(best_result_lines=bestResults,
                                          destination_file=ensemble_file + dataset_code,
                                          used_dataset=used_dataset)
    print("LabelsAndPredictions ottenute")

    print("Esecuzione ensembler")
    mv.ensembler(predALab=predALab, destination_file=ensemble_file + dataset_code)
    print("Esembler eseguito")

    end = time.time() - start

    print("Exec time with " + str(num_instaces) + " instance/s: " + str(end) + " seconds")


if __name__ == '__main__':
    dataset = ""

    #sys.argv.append("undersampled")

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

    runAnalysis(used_dataset, dataset_code)
