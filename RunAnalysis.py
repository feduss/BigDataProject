# coding=utf-8
from pyspark.shell import sc
import MajorityVote as mv
import time


def runAnalysis():

    used_dataset = 3

    analysis_file = "results"
    ensemble_file = "ensembles_metrics"
    dataset_code = "_u3"

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

    print("Exec time with " + str(num_instaces) + ": " + str(end))


if __name__ == '__main__':
    runAnalysis()
