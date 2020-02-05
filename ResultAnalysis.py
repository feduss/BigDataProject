from sklearn import metrics


def resultAnalisys(labelsAndPredictions, elementNumber):

    # Calcolo le quantità di true positives, true negatives, false positives e false negatives, dove:
    # True positives sono i record fraudolenti riconosciuti come tali
    # True negatives sono quelli legittimi riconosciuti come tali
    # False positives sono quelli fraudolenti riconosciuti come legittimi
    # False negatives son quelli legittimi etichettati come fraudolenti
    TP = labelsAndPredictions.filter(lambda v_p: v_p[0] == v_p[1] and v_p[0] == 1).count()
    TN = labelsAndPredictions.filter(lambda v_p: v_p[0] == v_p[1] and v_p[0] == 0).count()
    FP = labelsAndPredictions.filter(lambda v_p: v_p[0] != v_p[1] and v_p[0] == 1).count()
    FN = labelsAndPredictions.filter(lambda v_p: v_p[0] != v_p[1] and v_p[0] == 0).count()

    # Calcolo l'errore dividendo il numero di predizioni sbagliate per il numero di dati
    testErr = (FP + FN) / elementNumber

    # Sensitivity = transazioni fraudolente riconosciute come tali sul totale di record etichettati come fraudolenti
    sensitivity = TP / (TP + FN)
    # Fallout = transazioni fraudolente riconosciute come legittime sul totale delle legittime
    fallout = FP / (FP + TN)
    # Specificity = transazioni legittime riconosciute come tali sul totale delle legittime
    specificity = TN / (TN + FP)
    # Miss rate = transazioni legittime riconosciute come fraudolente sul totale delle fraudolente
    missRate = FN / (FN + TP)

    # Calcolo la curva di ROC:
    # y_true = etichette originali dei record
    # y_score = predizioni dei record
    # pos_label = valore della classe positiva nelle etichette valutate
    y_true = labelsAndPredictions.keys().collect()
    y_score = labelsAndPredictions.values().collect()
    (fpr, tpr, thresholds) = metrics.roc_curve(y_true=y_true, y_score=y_score, pos_label=1)

    # Calcolo dell'AUC, o Area Under ROC Curve
    AUC = metrics.auc(fpr, tpr)

    results = Results(sensitivity, fallout, specificity, missRate, AUC, testErr)

    # Stampe varie
    print('Record del test set = ' + str(labelsAndPredictions.count()))
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

    # print('Learned classification tree model:')
    # print(model.toDebugString())

    return results


class Results:
    def __init__(self, sensitivity, fallout, specificity, missRate, AUC, testError):
        self.sensitivity = float("{0:.4f}".format(sensitivity))
        self.fallout = float("{0:.4f}".format(fallout))
        self.specificity = float("{0:.4f}".format(specificity))
        self.missRate = float("{0:.4f}".format(missRate))
        self.AUC = float("{0:.4f}".format(AUC))
        self.testErr = float("{0:.4f}".format(testError))
