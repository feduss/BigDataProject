# BigDataProject
2019/20, Saccà Federico e Merli Silvio

GUIDA ALL'INSTALLAZIONE:

- Creare un utente IAM su aws, salvando access_key_id e secret_access_key_id, che serviranno in seguito:
    - Andare su aws e cliccare su Servizi
    - Cercare la voce IAM sotto Servizi, sicurezza, identità e conformità, e cliccarci
    - Cliccare su Utenti e, in seguito, Aggiungi Utente
    - Inserire uno Username a scelta
    - Scegliere Programmatic Access come access type e cliccare su next (in basso a dx)
    - Selezionare Attach Existing Policies Directly e flaggare Administrator Access
    - Cliccare next sino a quando non è possibile creare l'utente
    - Salvare le credenziali in formato csv
    
- Installare awscli con il seguente comando:
    
    ```console
    user@user:~$ sudo apt install awscli
    ```
 
- Dopo aver installato aws cli, eseguire il seguente comando per salvare le credenziali in locale:

    ```console
    user@user:~$ aws configure
    ```
    
    - Inserendo l'access_key_id e la secret_access_key_id salvate prima, eu-west-3 come region name e json come output format
    
- Creare una key-pair su aws, avendo cura di salvare la chiave.pem, che servirà in seguito.
    
    - Andare su aws e cliccare su Servizi,
    - Cliccare su EC2
    - Cliccare su Key Pair in Network & Security
    - Cliccare su creare Key Pair in alto a dx
    - Assegnare un nome a scelta e cliccare sul formato file PEM
    - Cliccare su crea e salvare la chiave PEM

- Scaricare lo zip "terraform.zip" e scompattarlo.
- Aggiornare il file delle variabili (terraform.tfvars della cartella appena estratta) con i dati richiesti (per un template di come impostare le variabili, controllare il file variables.tf). Il progetto è stato testato con istanze aws t2.2xlarge.

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


