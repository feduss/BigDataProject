import csv
from statistics import mean, stdev


def ResultAnalysis(classifiers, source_file, destination_file):
    # used_dataset = 2  # Dataset utilizzato per creare e testare i classificatori; valori: [1, 2]
    # classifiers = 6   # Da aggiornare con il numero di classificatori contenuti nel file

    with open('CSV_Results/' + source_file + '.csv', 'r') as metrics_reader:
        with open('CSV_Results/' + destination_file + '.csv', 'w') as result_file:
            csvWriter = csv.writer(result_file)

            # Recupero il moltiplicatore usato nell'analisi da valutare
            supp_string = metrics_reader.readline()
            multi = int(supp_string.split(" ")[1])

            for i in range(0, classifiers):  # Per ogni classificatore salvato
                supp_string = metrics_reader.readline().split(": ")

                # Recupero il nome del classificatore
                name = supp_string[0]

                # Recupero il numero di test effettuati per il singolo classificatore
                num_test = int(supp_string[1].split(" ")[0])

                # Recupero i nomi dei parametri del classificatore
                supp_string = metrics_reader.readline().split(",")
                param_names = supp_string[1:(len(supp_string)-7)]

                # Header del classificatore nei risultati
                csvWriter.writerow([name + ": " + str(num_test) + " versions x " + str(multi) + " iterations"])
                csvWriter.writerow(['Version', 'Sensitivity(Mean)', 'Sensitivity(StDev)',
                                    'Fallout(Mean)', 'Fallout(StDev)',
                                    'Specificity(Mean)', 'Specificity(StDev)',
                                    'Miss_Rate(Mean)', 'Miss_Rate(StDev)',
                                    'Test_Error(Mean)', 'Test_Error(StDev)',
                                    'AUC(Mean)', 'AUC(StDev)',
                                    'Exec_Time(Mean)', 'Exec_Time(StDev)'])

                average = []
                standDev = []
                params = []

                for j in range(0, num_test):  # Per ogni test con gli stessi parametri
                    sensitivity = []
                    fallout = []
                    specificity = []
                    miss_rate = []
                    test_error = []
                    AUC = []
                    exec_time = []

                    # Recupero le metriche delle singole iterazioni dello stesso test
                    for k in range(0, multi):
                        supp_string = metrics_reader.readline().split(",")
                        val = len(supp_string)

                        if k == 0:
                            params.append(supp_string[1:(len(param_names) + 1)])

                        sensitivity.append(float(supp_string[val-7]))
                        fallout.append(float(supp_string[val-6]))
                        specificity.append(float(supp_string[val-5]))
                        miss_rate.append(float(supp_string[val-4]))
                        test_error.append(float(supp_string[val-3]))
                        AUC.append(float(supp_string[val-2]))
                        exec_time.append(float(supp_string[val-1][:-1]))

                    # Calcolo la media e la deviazione standard delle metriche nell'iterazione
                    average.append({"Sensitivity": float("{0:.4f}".format(mean(sensitivity))),
                                    "Fallout": float("{0:.4f}".format(mean(fallout))),
                                    "Specificity": float("{0:.4f}".format(mean(specificity))),
                                    "Miss_Rate": float("{0:.4f}".format(mean(miss_rate))),
                                    "Test_Error": float("{0:.4f}".format(mean(test_error))),
                                    "AUC": float("{0:.4f}".format(mean(AUC))),
                                    "Exec_Time": float("{0:.4f}".format(mean(exec_time)))})

                    '''
                    standDev.append({"Sensitivity": float("{0:.4f}".format(stdev(sensitivity))),
                                     "Fallout": float("{0:.4f}".format(stdev(fallout))),
                                     "Specificity": float("{0:.4f}".format(stdev(specificity))),
                                     "Miss_Rate": float("{0:.4f}".format(stdev(miss_rate))),
                                     "Test_Error": float("{0:.4f}".format(stdev(test_error))),
                                     "AUC": float("{0:.4f}".format(stdev(AUC))),
                                     "Exec_Time": float("{0:.4f}".format(stdev(exec_time)))})
                    '''
                    standDev.append({"Sensitivity": 0.0,
                                     "Fallout": 0.0,
                                     "Specificity": 0.0,
                                     "Miss_Rate": 0.0,
                                     "Test_Error": 0.0,
                                     "AUC": 0.0,
                                     "Exec_Time": 0.0})

                    csvWriter.writerow([str(j+1), str(average[j]["Sensitivity"]), str(standDev[j]["Sensitivity"]),
                                        str(average[j]["Fallout"]), str(standDev[j]["Fallout"]),
                                        str(average[j]["Specificity"]), str(standDev[j]["Specificity"]),
                                        str(average[j]["Miss_Rate"]), str(standDev[j]["Miss_Rate"]),
                                        str(average[j]["Test_Error"]), str(standDev[j]["Test_Error"]),
                                        str(average[j]["AUC"]), str(standDev[j]["AUC"]),
                                        str(average[j]["Exec_Time"]), str(standDev[j]["Exec_Time"])])

                min_index = 0
                for j in range(1, len(average)):
                    if average[j]["Test_Error"] < average[min_index]["Test_Error"]:
                        min_index = j
                    elif average[j]["Test_Error"] == average[min_index]["Test_Error"]:
                        if average[j]["Exec_Time"] < average[min_index]["Exec_Time"]:
                            min_index = j

                csvWriter.writerow(["Best version of " + name + " is test n°" + str(min_index + 1)])

                csvWriter.writerow(["With " + str(len(param_names)) + " parameters:"])
                for j in range(len(param_names)):
                    csvWriter.writerow([param_names[j] + ": " + params[min_index-1][j]])

                csvWriter.writerow(["-"])

                csvWriter.writerow(["Mean and St_Dev of the whole " + name + " classifier metrics are as follows:"])

                csvWriter.writerow(
                    ["Sensitivity: Mean = "
                     + str(float("{0:.4f}".format(mean([average[j]["Sensitivity"] for j in range(len(average))]))))
                     + "; St_Dev = "
                     + str(float("{0:.4f}".format(stdev([average[j]["Sensitivity"] for j in range(len(standDev))]))))
                     ])

                csvWriter.writerow(
                    ["Fallout: Mean = "
                     + str(float("{0:.4f}".format(mean([average[j]["Fallout"] for j in range(len(average))]))))
                     + "; St_Dev = "
                     + str(float("{0:.4f}".format(stdev([average[j]["Fallout"] for j in range(len(standDev))]))))
                     ])

                csvWriter.writerow(
                    ["Specificity: Mean = "
                     + str(float("{0:.4f}".format(mean([average[j]["Specificity"] for j in range(len(average))]))))
                     + "; St_Dev = "
                     + str(float("{0:.4f}".format(stdev([average[j]["Specificity"] for j in range(len(standDev))]))))
                     ])

                csvWriter.writerow(
                    ["Miss_Rate: Mean = "
                     + str(float("{0:.4f}".format(mean([average[j]["Miss_Rate"] for j in range(len(average))]))))
                     + "; St_Dev = "
                     + str(float("{0:.4f}".format(stdev([average[j]["Miss_Rate"] for j in range(len(standDev))]))))
                     ])

                csvWriter.writerow(
                    ["Test_Error: Mean = "
                     + str(float("{0:.4f}".format(mean([average[j]["Test_Error"] for j in range(len(average))]))))
                     + "; St_Dev = "
                     + str(float("{0:.4f}".format(stdev([average[j]["Test_Error"] for j in range(len(standDev))]))))
                     ])

                csvWriter.writerow(
                    ["AUC: Mean = "
                     + str(float("{0:.4f}".format(mean([average[j]["AUC"] for j in range(len(average))]))))
                     + "; St_Dev = "
                     + str(float("{0:.4f}".format(stdev([average[j]["AUC"] for j in range(len(standDev))]))))
                     ])

                csvWriter.writerow(
                    ["Exec_Time: Mean = "
                     + str(float("{0:.4f}".format(mean([average[j]["Exec_Time"] for j in range(len(average))]))))
                     + "; St_Dev = "
                     + str(float("{0:.4f}".format(stdev([average[j]["Exec_Time"] for j in range(len(standDev))]))))
                     ])

                supp_string = metrics_reader.readline()
                csvWriter.writerow([supp_string[:-1]])
