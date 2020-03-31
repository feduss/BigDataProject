# BigDataProject
2019/20, Saccà Federico e Merli Silvio

Impostare python3 su pyspark:
```console
user@user:~$ sudo nano /spark/bin/pyspark
```
Ed impostare PYSPARK_PYTHON=python3


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
user@user:~$ ./spark/bin/spark-submit --master spark://IPMASTER:7077 /home/NOMEUTENTE/PycharmProjects/BigDataProject/RunTesting.py 
```

Per avviare solo l'analisi:
```console
user@user:~$ ./spark/bin/spark-submit --master spark://IPMASTER:7077 /home/NOEMUTENTE/PycharmProjects/BigDataProject/RunAnalysis.py 

