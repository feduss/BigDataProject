# coding=utf-8
import time

from pyspark.shell import sc

import TestClassifiers as tc
import ResultAnalysis as ra
import MajorityVote as mv
import RunAnalysis as ra


if __name__ == "__main__":
    verbose = False
    multiplier = 5
    classifiers = 5
    used_dataset = 3

    metric_file = "classifiers_metrics"
    analysis_file = "results"
    ensemble_file = "ensembles_metrics"
    dataset_code = "_u3"

    mainTime = time.time()
    tc.mainTestClassifier(destination_file=metric_file + dataset_code,
                          verbose=verbose, multiplier=multiplier, used_dataset=used_dataset)
    print("Test eseguiti in " + str(time.time() - mainTime) + " secondi")

    ra.ResultAnalysis(source_file=metric_file + dataset_code,
                      destination_file=analysis_file + dataset_code,
                      classifiers=classifiers)
    print("Analisi ultimate. File Results pronto")

    ra.runAnalysis()
