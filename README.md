# BigDataProject
2019/20, Saccà Federico e Merli Silvio

Comandi richiesti per eseguire il codice:
```console
user@user:~$ sudo pip3 install pyspark
user@user:~$ sudo pip3 install pandas
user@user:~$ sudo pip3 install sklearn
user@user:~$ sudo pip3 install statistics
```
Dalla root del progetto:

Per avviare il testing e l'analisi:
```console
user@user:~$ ./spark/bin/spark-submit --master spark://IPMASTER:7077 /home/NOMEUTENTE/PycharmProjects/BigDataProject/RunAnalysis.py 
```

Per avviare solo il testing:
```console
user@user:~$ ./spark/bin/spark-submit --master spark://IPMASTER:7077 /home/NOEMUTENTE/PycharmProjects/BigDataProject/RunTesting.py 

