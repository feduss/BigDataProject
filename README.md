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
    
    Cambiare poi i permessi della chiave.pem, digitando sul terminale aperto nella cartella della chiave (mettendo il nome della chiave):
    ```console
    user@user:~$ chmod 400 my-key-pair.pem
    ```

- Scaricare lo zip "terraform.zip" e scompattarlo.
- Aggiornare il file delle variabili (terraform.tfvars della cartella appena estratta) con i dati richiesti (per un template di come impostare le variabili, controllare il file variables.tf). Il progetto è stato testato con istanze aws t2.2xlarge. Il parametro di default per il tipo di istanze è impostato a "t2.micro"; per cambiarlo manualmente, aggiungere al file terraform.tfvars la voce "instance_type" e assegnare il tipo desiderato (stessa cosa se si volesse cambiare regione o ami per le istanze, rispettivamente nelle variabili "AWS_region" e "AMI_code").

- Aprire un terminale nella cartella appena estratta ed eseguire i seguenti comandi.
    
    - Per inizializzare terraform, scaricando i plugin dei provider (in questo caso aws) da utilizzare:
    ```console
    user@user:~$ ./terraform init
    ```
    
    - Per visualizzare ciò che terraform dovrà creare:
    ```console
    user@user:~$ ./terraform plan
    ```
  
    - Per avviare la creazione delle istanze:
    ```console
    user@user:~$ ./terraform apply -auto-approve
    ```
    
- Accedere alle istanze su AWS attraverso ssh (tasto dx sull'istanza aws e cliccare su connect, usando poi la riga di codice sotto "Example" per accedervi). In tutte le istanze, seguire questi passaggi:

    ```console
    user@user:~$ sudo nano /etc/hosts
    ```
    e modificare: 
    IP.MA.ST.ER con IP o DNS del master (namenode)
    IP.SL.AV.E1 con IP o DNS dello slave 1 (datanode1)
    IP.SL.AV.E2 con IP o DNS dello slave 2 (datanode2)
    IP.SL.AV.E3 con IP o DNS dello slave 3 (datanode3)
    !!! Non farsi confondere dal fatto che, nel file, IP.MA.ST.ER è assegnato sia a namenode che a datanode1. Il nodo principale crea comunque uno slave su cui lavorare, quindi i numeri dei nodi solo slave sono slittati di +1 !!!
    
- Solo nel master, quando tutte le istanze sono attive, lanciare i seguenti comandi, per ultimare la configurazione.

    Per far riconoscere gli slaves al master:
    ```console
    user@user:~$ ssh datanode2 'cat >> /home/ubuntu/.ssh/authorized_keys'< /home/ubuntu/.ssh/id_rsa.pub
    user@user:~$ ssh datanode3 'cat >> /home/ubuntu/.ssh/authorized_keys'< /home/ubuntu/.ssh/id_rsa.pub
    user@user:~$ ssh datanode4 'cat >> /home/ubuntu/.ssh/authorized_keys'< /home/ubuntu/.ssh/id_rsa.pub
    ```
    
    [OPZIONALE] Per rimuovere i log di info da spark (per rendere visibili i nostri)
    
    - Andare nella cartella spark/conf
    - Copiare il contenuto del file log4j.properties.template in un nuovo file log4j.properties
    - Scorrere sino "log4j.rootCategory=INFO, console" e digitare ERROR al posto di INFO
    

GUIDA ALL'ESECUZIONE DEL CODICE:

- Avviare il master da namenode (terminale nella home dell'utente corrente):

    ```console
    user@user:~$ ./spark/sbin/start-master.sh
    ```

- Avviare gli slave in namenode, datanode1 e gli altri datanode:

    ```console
    user@user:~$ ./spark/sbin/start-slave.sh spark://dnsmaster:7077
    ```
    !!! Nel caso di errori dovuti alla mancanza di java, installarlo manualmente con il seguente comando !!!
    
    ```console
    user@user:~$ sudo apt-get install -y openjdk-8-jdk
    ```

!!! Verificare che tutti gli slave siano stati riconosciuti dal master, controllando su IPMASTER:8080 nel browser !!!

Nel master:
    
- Per avviare il testing e l'analisi, da un terminale aperto nella home dell'utente corrente:
```console
user@user:~$ ./spark/bin/spark-submit --master spark://IPMASTER:7077 /home/ubuntu/BigDataProject/RunTesting.py <undersampled, normalized>
```

- Per avviare solo l'analisi:
```console
user@user:~$ ./spark/bin/spark-submit --master spark://IPMASTER:7077 /home/ubuntu/BigDataProject/RunAnalysis.py <undersampled, normalized>
```


