import time
import TestClassifiers as tc
import ResultAnalysis as ra
import MajorityVote as mv

if __name__ == "__main__":
    verbose = False
    multiplier = 5
    classifiers = 5

    metric_file = "classifiers_metrics"
    analysis_file = "results"
    ensemble_file = "ensembles_metrics"
    dataset_code = "_u1"

    mainTime = time.time()
    tc.mainTestClassifier(destination_file=metric_file + dataset_code,
                          verbose=verbose, multiplier=multiplier, used_dataset=1)
    print("Test eseguiti in " + str(time.time() - mainTime) + " secondi")

    ra.ResultAnalysis(source_file=metric_file + dataset_code,
                      destination_file=analysis_file + dataset_code,
                      classifiers=classifiers)
    print("Analisi ultimate. File Results pronto")

    bestResults = mv.getBestResults(source_file=analysis_file + dataset_code,
                                    num_classifiers=5)

    predALab = mv.getLabelsAndPredictions(best_result_lines=bestResults,
                                          destination_file=ensemble_file + dataset_code)

    mv.ensembler(predALab=predALab, destination_file=ensemble_file + dataset_code)
