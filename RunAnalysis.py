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

    for i in range(1, 2):
        if i == 1:              # dataset = creditcard_undersampled1
            dataset_code = "_u1"
        elif i == 2:            # dataset = creditcard_undersampled2
            dataset_code = "_u2"
        elif i == 3:            # dataset = creditcard_normalized1
            dataset_code = "_n1"
        elif i == 4:            # dataset = creditcard_normalized2
            dataset_code = "_n2"
        '''
        mainTime = time.time()
        tc.mainTestClassifier(destination_file=metric_file + dataset_code,
                              verbose=verbose, multiplier=multiplier, used_dataset=1)
        print("Test col dataset n° " + str(i) + " eseguiti in " + str(time.time() - mainTime) + " secondi")

        ra.ResultAnalysis(source_file=metric_file + dataset_code,
                          destination_file=analysis_file + dataset_code,
                          classifiers=classifiers)
        print("Analisi ultimate. File Results pronto")
        '''
        bestResults = mv.getBestResults(source_file=analysis_file + dataset_code,
                                        num_classifiers=5)

        predALab = mv.getLabelsAndPredictions(best_result_lines=bestResults,
                                              destination_file=ensemble_file + dataset_code)

        mv.ensembler(predALab=predALab, destination_file=ensemble_file + dataset_code)
