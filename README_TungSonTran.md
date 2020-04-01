This project is based on the source code provided by C.Baziotis et. al.,
My own code is located in the folder "./model/task1"

In order for the code to function, must download word embedding file from  
https://drive.google.com/open?id=11zrXc1h_saJsMT6eo0VARKeZuzvK2bU0, 
and rename as "word2vec_300_6_concatened.txt"

The experiment can be started by running the following command 

```bash
python .\model\task1\base_experiment.py 
```

The above command must be specified with parameters provided as follow:
  -h, --help           show help message and exit
  --clf CLF
                       (Required) Choose one of the following schemes for task EI-oc, V-oc
                       LoR (Logistic Regressor)
                       SVC (Support Vector Classifier)
  --reg REG
                       (Required) Choose one of the following schemes for task EI-reg, V-reg
                       SGDR (Stochastic Gradient Descent Regressor)
                       SVR (Support Vector Regressor)
                       RFR (Random Forest Regressor)
  --task TASK
                       (Optional) Run a specific subtask. Suitable for demo
                       EI_reg
                       EI_oc
                       V_reg
                       V_oc
  --finetune FINETUNE  
                       (Optional) Enable tuning if set to "true". Requires lots of training time
  --baseline BASELINE  
                       (Optional) Run the baseline BOW model if set to "true". If not specified, will run the NBOW model instead.

In case stdout is filled, add ">log.txt" after the bash command to export log to a file.
The output of the experiment is "./out/experiments/Text Mining Project.json"


