# BigDataProject
2019/20, Saccà Federico e Merli Silvio

GUIDA ALL'INSTALLAZIONE:

-Caricamento repository e dataset-

Effettuare la clone con git:
```console
user@user:~$ git lfs https://github.com/feduss/BigDataProject.git
```

Scaricare il dataset da google drive:
```console
user@user:~$ wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1wr2DC3jRIwtmd-_kt-kbskDNgFHL0RBS' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1wr2DC3jRIwtmd-_kt-kbskDNgFHL0RBS" -O creditcard.csv && rm -rf /tmp/cookies.txt
```

<INSERIRE QUI GUIDA TERRAFORM/PYSPARK/HADOOP>

-Installazione requisiti-

Nelle istanze aws, seguire questi passaggi:

Impostare python3 su pyspark:
```console
user@user:~$ sudo nano spark/conf/spark-env.sh
```
Ed aggiungere in coda:
```bash
export PYSPARK_PYTHON=python3
```

Pacchetti richiesti per eseguire il codice:
```console
user@user:~$ sudo pip3 install pyspark
user@user:~$ sudo pip3 install pandas
user@user:~$ sudo pip3 install sklearn
user@user:~$ sudo pip3 install statistics
```

-Esecuzione del codice-

Per avviare il testing e l'analisi:
```console
user@user:~$ ./spark/bin/spark-submit --master spark://IPMASTER:7077 /home/NOMEUTENTE/BigDataProject/RunTesting.py <undersampled, normalized>
```

Per avviare solo l'analisi:
```console
user@user:~$ ./spark/bin/spark-submit --master spark://IPMASTER:7077 /home/NOMEMUTENTE/BigDataProject/RunAnalysis.py <undersampled, normalized>
```


