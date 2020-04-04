# coding=utf-8
from pyspark.mllib.evaluation import BinaryClassificationMetrics as metrics


def metricsEvalutation(predictionsAndLabels, elementNumber, verbose):
    # Calcolo le quantità di true positives, true negatives, false positives e false negatives, dove:
    # True positives sono i record fraudolenti riconosciuti come tali
    # True negatives sono quelli legittimi riconosciuti come tali
    # False positives sono quelli fraudolenti riconosciuti come legittimi
    # False negatives son quelli legittimi etichettati come fraudolenti

    TPdata = predictionsAndLabels.filter(lambda v_p: v_p[0] == v_p[1] and v_p[1] == 1)
    if not TPdata.isEmpty():
        TP = TPdata.count()
    else:
        TP = 0

    TNdata = predictionsAndLabels.filter(lambda v_p: v_p[0] == v_p[1] and v_p[1] == 0)
    if not TNdata.isEmpty():
        TN = TNdata.count()
    else:
        TN = 0

    FPdata = predictionsAndLabels.filter(lambda v_p: v_p[0] != v_p[1] and v_p[1] == 1)
    if not FPdata.isEmpty():
        FP = FPdata.count()
    else:
        FP = 0

    FNdata = predictionsAndLabels.filter(lambda v_p: v_p[0] != v_p[1] and v_p[1] == 0)
    if not FNdata.isEmpty():
        FN = FNdata.count()
    else:
        FN = 0

    # Calcolo l'errore dividendo il numero di predizioni sbagliate per il numero di dati
    testErr = (FP + FN) / elementNumber

    # Sensitivity = transazioni fraudolente riconosciute come tali sul totale di record etichettati come fraudolenti
    sensitivity = (0 if (TP + FN) == 0 else TP / (TP + FN))

    # Fallout = transazioni fraudolente riconosciute come legittime sul totale delle legittime
    fallout = (0 if (FP + TN) == 0 else FP / (FP + TN))

    # Specificity = transazioni legittime riconosciute come tali sul totale delle legittime
    specificity = (0 if (TN + FP) == 0 else TN / (TN + FP))

    # Miss rate = transazioni legittime riconosciute come fraudolente sul totale delle fraudolente
    missRate = (0 if (FN + TP) == 0 else FN / (FN + TP))

    # Calcolo la curva di ROC
    PaL = predictionsAndLabels.map(lambda x: (x[0], x[1]))
    AUC = metrics(PaL).areaUnderROC

    results = Results(sensitivity, fallout, specificity, missRate, AUC, testErr)

    # Stampe varie
    if verbose:
        print('Record del test set = ' + str(PaL.count()))
        print('Totali = ' + str(TP + TN + FP + FN) + '\n')

        print('True positives = ' + str(TP))
        print('True negatives = ' + str(TN))
        print('False positives = ' + str(FP))
        print('False negatives = ' + str(FP) + '\n')

        print('Sensitivity = ' + str(results.sensitivity * 100) + '%')
        print('Miss rate = ' + str(results.missRate * 100) + '%\n')

        print('Specificity = ' + str(results.specificity * 100) + '%')
        print('Fallout = ' + str(results.fallout * 100) + '%\n')

        print('AUC = ' + str(results.AUC * 100) + '%')
        print('Test Error = ' + str(results.testErr * 100) + '%')
    return results


class Results:
    def __init__(self, sensitivity, fallout, specificity, missRate, AUC, testError):
        self.sensitivity = float("{0:.4f}".format(sensitivity))
        self.fallout = float("{0:.4f}".format(fallout))
        self.specificity = float("{0:.4f}".format(specificity))
        self.missRate = float("{0:.4f}".format(missRate))
        self.AUC = float("{0:.4f}".format(AUC))
        self.testErr = float("{0:.4f}".format(testError))
