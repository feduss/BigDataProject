# BigDataProject
2019/20, Saccà Federico e Merli Silvio

GUIDA ALL'INSTALLAZIONE:

Installare git lfs:
```console
user@user:~$ wget https://github.com/git-lfs/git-lfs/releases/download/v2.10.0/git-lfs-linux-amd64-v2.10.0.tar.gz
user@user:~$ mkdir git-lfs-linux-amd64-v2.10.0
user@user:~$ tar -xf git-lfs-linux-amd64-v2.10.0.tar.gz --directory /home/ubuntu/git-lfs-linux-amd64-v2.10.0/
user@user:~$ sudo ./git-lfs-linux-amd64-v2.10.0/install.sh
```

Effettuare la clone con git lfs:
```console
user@user:~$ git lfs clone https://github.com/feduss/BigDataProject.git
```

PER ESEGUIRE QUESTO PROGETTO E' NECESSARIO AVERE, CORRETTAMENTE INSTALLATO, PYSPARK.

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

AVVIARE IL MASTER E GLI SLAVES CON SPARK

Per avviare il testing e l'analisi:
```console
user@user:~$ ./spark/bin/spark-submit --master spark://IPMASTER:7077 /home/NOMEUTENTE/BigDataProject/RunTesting.py 
```

Per avviare solo l'analisi:
```console
user@user:~$ ./spark/bin/spark-submit --master spark://IPMASTER:7077 /home/NOMEMUTENTE/BigDataProject/RunAnalysis.py 

