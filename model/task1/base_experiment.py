#############################################################################################
## Author: Tung Son Tran
## Modified after C.Baziotis et. al.,
############################################################################################# 

import json
import os
import sys
import argparse

sys.path.append('.')
sys.path.append('..')
sys.path.append('../../')
sys.path.append('../../../')

from collections import defaultdict

from config import BASE_PATH, EXPS_PATH
from model.task1.baseline_models import train_ei_reg, train_ei_oc, train_v_reg, train_v_oc
from modules.sklearn.models import nbow_model, bow_model, eval_reg, eval_mclf
from utils.load_embeddings import load_word_vectors
from utils.nlp import twitter_preprocess

help_reg="""
(Required) Choose one of the following schemes for task EI-reg, V-reg
SGDR (Stochastic Gradient Descent Regressor)
SVR (Support Vector Regressor)
RFR (Random Forest Regressor)
"""
help_clf="""
(Required) Choose one of the following schemes for task EI-oc, V-oc
LoR (Logistic Regressor)
SVC (Support Vector Classifier)
"""
help_task="""
(Optional) Run a specific subtask. Suitable for demo
EI_reg
EI_oc
V_reg
V_oc
"""
help_finetune="""
(Optional) Enable tuning if set to "true". Requires lots of training time
"""
help_baseline="""
(Optional) Run the baseline model if set to "true". Faster but worse performance
"""

# Arguments parsing
parser = argparse.ArgumentParser(description='test',
                                 formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('--clf', help=help_clf)
parser.add_argument('--reg', help=help_reg)
parser.add_argument('--task', help=help_task)
parser.add_argument('--finetune', help=help_finetune)
parser.add_argument('--baseline', help=help_baseline)
args = parser.parse_args()
print(args)
if args.clf is None and args.reg is None:
    raise ValueError("Please specify classification and regression algorithms!") 

# Load embeddings
emb_files = [
    ("word2vec_300_6_concatened.txt", 310)
]
embeddings = {}
for e, d in emb_files:
    file = os.path.join(BASE_PATH, "embeddings", e)
    word2idx, idx2word, weights = load_word_vectors(file, d)
    embeddings[e.split(".")[0]] = (weights, word2idx)

# Models config
if args.clf is not None:
    bow_clf = bow_model(args.clf)
    nbow_clf = {"nbow_{}".format(name): nbow_model(args.clf, e, w2i)
            for name, (e, w2i) in embeddings.items()}
if args.reg is not None:
    bow_reg = bow_model(args.reg)
    nbow_reg = {"nbow_{}".format(name): nbow_model(args.reg, e, w2i)
            for name, (e, w2i) in embeddings.items()}

# Preprocess
preprocessor = twitter_preprocess()

results = defaultdict(dict)
# ###########################################################################
# # 1. Task EI-reg: Detecting Emotion Intensity (regression)
# ###########################################################################
if args.task == 'EI_reg':
    for emotion in ["joy", "sadness", "fear", "anger"]:
        task = "EI-reg:{}".format(emotion)
        if args.baseline == 'true':
            print("Running task {}-{} using baseline model".format(args.task,emotion))
            dev, gold = train_ei_reg(emotion=emotion, 
                                    model=bow_reg, 
                                    algorithm=args.reg,
                                    evaluation=eval_reg, 
                                    finetune=args.finetune,
                                    baseline=args.baseline,
                                    preprocessor=preprocessor)
            results[task]["bow"] = {"dev": dev, "gold": gold}
        else:
            print("Running task {}-{} using NN model".format(args.task,emotion))
            for name, model in nbow_reg.items():
                dev, gold = train_ei_reg(emotion=emotion,
                                        model=model,
                                        algorithm=args.reg,
                                        evaluation=eval_reg,
                                        finetune=args.finetune,
                                        baseline=args.baseline,
                                        preprocessor=preprocessor)
                results[task][name] = {"dev": dev, "gold": gold}

###########################################################################
# 2. Task V-reg: Detecting Valence or Sentiment Intensity (regression)
###########################################################################
elif args.task == 'V_reg':
    task = "V-reg"
    if args.baseline == 'true':
        print("Running task {} using baseline model".format(args.task))
        dev, gold = train_v_reg(model=bow_reg,
                                algorithm=args.reg,
                                evaluation=eval_reg,
                                finetune=args.finetune,
                                baseline=args.baseline,
                                preprocessor=preprocessor)
        results[task]["bow"] = {"dev": dev, "gold": gold}
    else:
        print("Running task {} using NN model".format(args.task))
        for name, model in nbow_reg.items():
            dev, gold = train_v_reg(model=model,
                                    algorithm=args.reg,
                                    evaluation=eval_reg,
                                    finetune=args.finetune,
                                    baseline=args.baseline,
                                    preprocessor=preprocessor)
            results[task][name] = {"dev": dev, "gold": gold}

###########################################################################
# 3. Task EI-oc: Detecting Emotion Intensity (ordinal classification)
###########################################################################
elif args.task == 'EI_oc':
    for emotion in ["joy", "sadness", "fear", "anger"]:
        task = "EI-oc:{}".format(emotion)
        if args.baseline == 'true':
            print("Running task {}-{} using baseline model".format(args.task,emotion))
            dev, gold = train_ei_oc(emotion=emotion,
                                    model=bow_clf,
                                    algorithm=args.clf,
                                    evaluation=eval_reg,
                                    finetune=args.finetune,
                                    baseline=args.baseline,
                                    preprocessor=preprocessor)
            results[task]["bow"] = {"dev": dev, "gold": gold}
        else:
            print("Running task {}-{} using NN model".format(args.task,emotion))
            for name, model in nbow_clf.items():
                dev, gold = train_ei_oc(emotion=emotion,
                                        model=model,
                                        algorithm=args.clf,
                                        evaluation=eval_reg,
                                        finetune=args.finetune,                                        
                                        baseline=args.baseline,
                                        preprocessor=preprocessor)
                results[task][name] = {"dev": dev, "gold": gold}

##########################################################################
# 4. Task V-oc: Detecting Valence (ordinal classification)
##########################################################################
elif args.task == 'V_oc':
    task = "V-oc"
    if args.baseline == 'true':
        print("Running task {} using baseline model".format(args.task))
        dev, gold = train_v_oc(model=bow_clf, 
                            algorithm=args.clf,
                            evaluation=eval_reg,
                            finetune=args.finetune,
                            baseline=args.baseline,
                            preprocessor=preprocessor)
        results[task]["bow"] = {"dev": dev, "gold": gold}
    else:
        print("Running task {} using NN model".format(args.task))
        for name, model in nbow_clf.items():
            dev, gold = train_v_oc(model=model, 
                                algorithm=args.clf,
                                evaluation=eval_reg,
                                finetune=args.finetune,
                                baseline=args.baseline,
                                preprocessor=preprocessor)
            results[task][name] = {"dev": dev, "gold": gold}

##########################################################################
with open(os.path.join(EXPS_PATH, "Text Mining Project.json"), 'w') as f:
    json.dump(results, f)
