# BigDataProject
2019/20, Saccà Federico e Merli Silvio

GUIDA ALL'INSTALLAZIONE:

- Scaricare lo zip "terraform.zip" e scompattarlo.
- Aggiornare il file delle variabili (terraform.tfvars nella root della cartella appena estratta) con i dati richiesti (ps: il codice del progetto è stato testato con istanza t2.2xlarge)

- Aprire un terminale nella cartella appena estratta ed eseguire i seguenti comandi.

    - Per visualizzare ciò che terraform dovrà creare:
    ```console
    user@user:~$ ./terraform plan
    ```
    
    - Per inizializzare terraform, scaricando i plugin dei provider (in questo caso aws) da utilizzare:
    ```console
    user@user:~$ ./terraform init
    ```
  
    - Per avviare la creazione delle istanze:
    ```console
    user@user:~$ ./terraform apply -auto-approve
    ```
    
- In tutte le istanze aws, seguire questi passaggi:

    ```console
    user@user:~$ sudo nano /etc/hosts
    ```
    e modificare:
    
    IP.MA.ST.ER
    IP.SL.AV.E1 
    IP.SL.AV.E2 
    IP.SL.AV.E3
    
    dove master è il namenode, slave1 è datanode1, etc.
    
- Solo nel master, quando tutte le istanze sono attive, lanciare i seguenti comandi, per ultimare la configurazione.

    Per far riconoscere gli slaves al master:
    ```console
    user@user:~$ ssh datanode2 'cat >> /home/ubuntu/.ssh/authorized_keys'< /home/ubuntu/.ssh/id_rsa.pub
    user@user:~$ ssh datanode3 'cat >> /home/ubuntu/.ssh/authorized_keys'< /home/ubuntu/.ssh/id_rsa.pub
    user@user:~$ ssh datanode4 'cat >> /home/ubuntu/.ssh/authorized_keys'< /home/ubuntu/.ssh/id_rsa.pub
    ```
    
    Per inizializzare i servizi di hadoop:
    
    ```console
    user@user:~$ hdfs namenode -format
    user@user:~$ $HADOOP_HOME/sbin/start-dfs.sh
    user@user:~$ $HADOOP_HOME/sbin/start-yarn.sh
    user@user:~$ $HADOOP_HOME/sbin/mr-jobhistory-daemon.sh start historyserver
    ```

GUIDA ALL'ESECUZIONE DEL CODICE:

- Avviare il master da namenode (terminale nella home dell'utente corrente):

    ```console
    user@user:~$ ./spark/sbin/start-master.sh
    ```

- Avviare gli slave in namenode, datanode1 e gli altri datanode:

    ```console
    user@user:~$ ./spark/sbin/start-slave.sh spark://dnsmaster:7077
    ```

!!!Verificare che tutti gli slave siano stati riconosciuti dal master, controllando su IPMASTER:8080 nel browser!!!
    
- Per avviare il testing e l'analisi, da un terminale aperto nella home dell'utente corrente:
```console
user@user:~$ ./spark/bin/spark-submit --master spark://IPMASTER:7077 /home/NOMEUTENTE/BigDataProject/RunTesting.py <undersampled, normalized>
```

- Per avviare solo l'analisi:
```console
user@user:~$ ./spark/bin/spark-submit --master spark://IPMASTER:7077 /home/NOMEMUTENTE/BigDataProject/RunAnalysis.py <undersampled, normalized>
```


