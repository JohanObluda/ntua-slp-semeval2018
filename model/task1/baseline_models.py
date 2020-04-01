#############################################################################################
## Author: Tung Son Tran
############################################################################################# 

import numpy

from dataloaders.task1 import parse
from model.params import TASK1_VREG, TASK1_EIOC, TASK1_EC, \
    TASK1_EIREG, TASK1_VOC
from utils.nlp import twitter_preprocess
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import pearsonr

def fit_function(model, evaluation, X_train, y_train, X_dev, y_dev, X_test, y_test, params, res_dev_list, res_test_list, params_list):
    model.fit(X_train, y_train)
    res_dev_temp = evaluation(model.predict(X_dev), y_dev)
    res_test_temp = evaluation(model.predict(X_test), y_test)
    res_dev_list.append(res_dev_temp)
    res_test_list.append(res_test_temp)
    params_list.append(params)

    return res_dev_list, res_test_list, params_list

def train_ei_reg(emotion, model, algorithm, evaluation, finetune, baseline, preprocessor=None):
    """
    1. Task EI-reg: Detecting Emotion Intensity (regression)

    Given:

        - a tweet
        - an emotion E (anger, fear, joy, or sadness)

    Task: determine the  intensity of E that best represents the mental state of
    the tweeter—a real-valued score between 0 and 1:

        - a score of 1: highest amount of E can be inferred
        - a score of 0: lowest amount of E can be inferred

    For each language: 4 training sets and 4 test sets: one for each emotion E.

    (Note that the absolute scores have no inherent meaning --
    they are used only as a means to convey that the instances
    with higher scores correspond to a greater degree of E
    than instances with lower scores.)

    :param emotion: emotions = ["anger", "fear", "joy", "sadness"]
    :param pretrained:
    :param finetune:
    :param unfreeze:
    :return:
    """

    if preprocessor is None:
        preprocessor = twitter_preprocess()

    model_config = TASK1_EIREG

    X_train, y_train = parse(task='EI-reg', emotion=emotion, dataset="train")
    X_dev, y_dev = parse(task='EI-reg', emotion=emotion, dataset="dev")
    X_test, y_test = parse(task='EI-reg', emotion=emotion, dataset="gold")

    # keep only scores
    y_train = [y[1] for y in y_train]
    y_dev = [y[1] for y in y_dev]
    y_test = [y[1] for y in y_test]

    name = model_config["name"] + "_" + emotion

    X_train = preprocessor("{}_{}".format(name, "train"), X_train)
    X_dev = preprocessor("{}_{}".format(name, "dev"), X_dev)
    X_test = preprocessor("{}_{}".format(name, "test"), X_test)
    
    params = []
    params_list = []
    res_dev_list = []
    res_test_list = []
    
    if algorithm == 'SVR':
        if finetune == 'true':
            for SVR_C in numpy.arange(10,50,1)/10:
                for SVR_gamma in numpy.arange(65,500,10)/100:
                    params = (SVR_C,SVR_gamma)
                    print("Now training with parameters: C: {}, Gamma: {}".format(SVR_C,SVR_gamma))
                    model.set_params(clf__C=SVR_C,clf__gamma=SVR_gamma)
                    fit = fit_function(model, evaluation, X_train, y_train, X_dev, y_dev, X_test, y_test, params, res_dev_list, res_test_list, params_list)
            print("Best result on gold set: ", max(fit[1], key=lambda x:x["pearson"]))
            print("Best params: ", fit[2][res_test_list.index(max(res_test_list, key=lambda x:x["pearson"]))])
        else:
            if emotion == 'joy' or emotion == 'sadness' or emotion == 'fear':
                SVR_C = 4.7
                SVR_gamma = 0.017
            elif emotion == 'anger':
                SVR_C = 1.6
                SVR_gamma = 0.083
            params = (SVR_C,SVR_gamma)
            model.set_params(clf__C=SVR_C,clf__gamma=SVR_gamma)
            fit = fit_function(model, evaluation, X_train, y_train, X_dev, y_dev, X_test, y_test, params, res_dev_list, res_test_list, params_list)
        
    elif algorithm == 'SGDR':
        if finetune == 'true':
            for SGDR_alpha in numpy.arange(1, 100)/1e05:
                params = (SGDR_alpha)
                print("Now training with parameters: alpha: {}".format(SGDR_alpha))
                model.set_params(clf__alpha=SGDR_alpha,clf__tol=1e-04,clf__n_iter=1000,clf__penalty='l2',clf__loss='squared_loss')
                fit = fit_function(model, evaluation, X_train, y_train, X_dev, y_dev, X_test, y_test, params, res_dev_list, res_test_list, params_list)
            print("Best result on gold set: ", max(fit[1], key=lambda x:x["pearson"]))
            print("Best params: ", fit[2][res_test_list.index(max(res_test_list, key=lambda x:x["pearson"]))])
        else:
            if emotion == 'joy' or emotion == 'fear' or emotion == 'anger':
                SGDR_alpha = 1e-05
            elif emotion == 'sadness':
                SGDR_alpha = 2.2e-04
            params = (SGDR_alpha)
            model.set_params(clf__alpha=SGDR_alpha,clf__tol=1e-04,clf__n_iter=1000,clf__penalty='l2',clf__loss='squared_loss')
            fit = fit_function(model, evaluation, X_train, y_train, X_dev, y_dev, X_test, y_test, params, res_dev_list, res_test_list, params_list)
    
    elif algorithm == 'RFR':
        if finetune == 'true':
            for RFR_n_estimators in numpy.arange(300,1000,100):
                for RFR_max_depth in numpy.arange(50,120,10):
                    for RFR_min_samples_split in [2,3,4]:
                        for RFR_min_samples_leaf in [2,3,4]:
                            params = (RFR_n_estimators,RFR_max_depth,RFR_min_samples_split,RFR_min_samples_leaf)
                            print("Now training with parameters: n_estimator: {}, max_depth: {}, min split: {}, min leaf: {}".format(RFR_n_estimators,RFR_max_depth,RFR_min_samples_split,RFR_min_samples_leaf))
                            model.set_params(clf__n_estimators=RFR_n_estimators,
                                             clf__max_depth=RFR_max_depth,
                                             clf__min_samples_split=RFR_min_samples_split,
                                             clf__min_samples_leaf=RFR_min_samples_leaf,
                                             clf__max_features='sqrt',
                                             clf__bootstrap=False,
                                             clf__n_jobs=-1,
                                             clf__tol=1e-04,clf__n_iter=1000,clf__penalty='l2',clf__loss='squared_loss')
                            fit = fit_function(model, evaluation, X_train, y_train, X_dev, y_dev, X_test, y_test, params, res_dev_list, res_test_list, params_list)
            print("Best result on gold set: ", max(fit[1], key=lambda x:x["pearson"]))
            print("Best params: ", fit[2][res_test_list.index(max(res_test_list, key=lambda x:x["pearson"]))])
        else:
            if emotion == 'joy':
                RFR_n_estimators = 600
                RFR_max_depth = 60
                RFR_min_samples_split = 2
            elif emotion == 'sadness':
                RFR_n_estimators = 600
                RFR_max_depth = 80
                RFR_min_samples_split = 2
            elif emotion == 'fear':
                RFR_n_estimators = 900
                RFR_max_depth = 90
                RFR_min_samples_split = 2
            elif emotion == 'anger':
                RFR_n_estimators = 900
                RFR_max_depth = 100
                RFR_min_samples_split = 3
            RFR_min_samples_leaf = 2
            params = (RFR_n_estimators,RFR_max_depth,RFR_min_samples_split,RFR_min_samples_leaf)
            model.set_params(clf__n_estimators=RFR_n_estimators,
                             clf__max_depth=RFR_max_depth,
                             clf__min_samples_split=RFR_min_samples_split,
                             clf__min_samples_leaf=RFR_min_samples_leaf,
                             clf__max_features='sqrt',
                             clf__bootstrap=False,
                             clf__n_jobs=-1,
                             clf__tol=1e-04,clf__n_iter=1000,clf__penalty='l2',clf__loss='squared_loss')
            fit = fit_function(model, evaluation, X_train, y_train, X_dev, y_dev, X_test, y_test, params, res_dev_list, res_test_list, params_list)

    res_dev = fit[0][fit[1].index(max(fit[1], key=lambda x:x["pearson"]))]
    res_test = max(fit[1], key=lambda x:x["pearson"])   

    return res_dev, res_test


def train_v_reg(model, algorithm, evaluation, finetune, baseline, preprocessor=None):
    """
    3. Task V-reg: Detecting Valence or Sentiment Intensity (regression)

    Given:
     - a tweet

    Task: determine the intensity of sentiment or valence (V)
    that best represents the mental state of the tweeter—a real-valued score
    between 0 and 1:

        a score of 1: most positive mental state can be inferred
        a score of 0: most negative mental state can be inferred

    For each language: 1 training set, 1 test set.

    (Note that the absolute scores have no inherent meaning --
    they are used only as a means to convey that the instances
    with higher scores correspond to a greater degree of positive sentiment
    than instances with lower scores.)

    :param pretrained:
    :param finetune:
    :param unfreeze:
    :return:
    """
    model_config = TASK1_VREG
    # load the dataset and split it in train and val sets
    X_train, y_train = parse(task='V-reg', dataset="train")
    X_dev, y_dev = parse(task='V-reg', dataset="dev")
    X_test, y_test = parse(task='V-reg', dataset="gold")

    # keep only scores
    y_train = [y[1] for y in y_train]
    y_dev = [y[1] for y in y_dev]
    y_test = [y[1] for y in y_test]

    name = model_config["name"]

    X_train = preprocessor("{}_{}".format(name, "train"), X_train)
    X_dev = preprocessor("{}_{}".format(name, "dev"), X_dev)
    X_test = preprocessor("{}_{}".format(name, "test"), X_test)

    params = []
    params_list = []
    res_dev_list = []
    res_test_list = []

    if algorithm == 'SVR':
        if finetune == 'true':
            for SVR_C in numpy.arange(10,50,1)/10:
                for SVR_gamma in numpy.arange(65,500,10)/100:
                    params = (SVR_C,SVR_gamma)
                    print("Now training with parameters: C: {}, Gamma: {}".format(SVR_C,SVR_gamma))
                    model.set_params(clf__C=SVR_C,clf__gamma=SVR_gamma)
                    fit = fit_function(model, evaluation, X_train, y_train, X_dev, y_dev, X_test, y_test, params, res_dev_list, res_test_list, params_list)
            print("Best result on gold set: ", max(fit[1], key=lambda x:x["pearson"]))
            print("Best params: ", fit[2][res_test_list.index(max(res_test_list, key=lambda x:x["pearson"]))])
        else:
            SVR_C = 2.6
            SVR_gamma = 0.085
            params = (SVR_C,SVR_gamma)
            model.set_params(clf__C=SVR_C,clf__gamma=SVR_gamma)
            fit = fit_function(model, evaluation, X_train, y_train, X_dev, y_dev, X_test, y_test, params, res_dev_list, res_test_list, params_list)
    
    elif algorithm == 'SGDR':
        if finetune == 'true':
            for SGDR_alpha in numpy.arange(1, 100)/1e05:
                params = (SGDR_alpha)
                print("Now training with parameters: alpha: {}".format(SGDR_alpha))
                model.set_params(clf__alpha=SGDR_alpha,clf__tol=1e-04,clf__n_iter=1000,clf__penalty='l2',clf__loss='squared_loss')
                fit = fit_function(model, evaluation, X_train, y_train, X_dev, y_dev, X_test, y_test, params, res_dev_list, res_test_list, params_list)
            print("Best result on gold set: ", max(fit[1], key=lambda x:x["pearson"]))
            print("Best params: ", fit[2][res_test_list.index(max(res_test_list, key=lambda x:x["pearson"]))])
        else:
            SGDR_alpha = 1e-05
            params = (SGDR_alpha)
            model.set_params(clf__alpha=SGDR_alpha,clf__tol=1e-04,clf__n_iter=1000,clf__penalty='l2',clf__loss='squared_loss')
            fit = fit_function(model, evaluation, X_train, y_train, X_dev, y_dev, X_test, y_test, params, res_dev_list, res_test_list, params_list)
    
    elif algorithm == 'RFR':
        if finetune == 'true':
            for RFR_n_estimators in numpy.arange(300,1000,100):
                for RFR_max_depth in numpy.arange(50,120,10):
                    for RFR_min_samples_split in [2,3,4]:
                        for RFR_min_samples_leaf in [2,3,4]:
                            params = (RFR_n_estimators,RFR_max_depth,RFR_min_samples_split,RFR_min_samples_leaf)
                            print("Now training with parameters: n_estimator: {}, max_depth: {}, min split: {}, min leaf: {}".format(RFR_n_estimators,RFR_max_depth,RFR_min_samples_split,RFR_min_samples_leaf))
                            model.set_params(clf__n_estimators=RFR_n_estimators,
                                             clf__max_depth=RFR_max_depth,
                                             clf__min_samples_split=RFR_min_samples_split,
                                             clf__min_samples_leaf=RFR_min_samples_leaf,
                                             clf__max_features='sqrt',
                                             clf__bootstrap=False,
                                             clf__n_jobs=-1,
                                             clf__tol=1e-04,clf__n_iter=1000,clf__penalty='l2',clf__loss='squared_loss')
                            fit = fit_function(model, evaluation, X_train, y_train, X_dev, y_dev, X_test, y_test, params, res_dev_list, res_test_list, params_list)
            print("Best result on gold set: ", max(fit[1], key=lambda x:x["pearson"]))
            print("Best params: ", fit[2][res_test_list.index(max(res_test_list, key=lambda x:x["pearson"]))])
        else:
            RFR_n_estimators = 400
            RFR_max_depth = 90
            RFR_min_samples_split = 2
            RFR_min_samples_leaf = 2
            params = (RFR_n_estimators,RFR_max_depth,RFR_min_samples_split,RFR_min_samples_leaf)
            model.set_params(clf__n_estimators=RFR_n_estimators,
                             clf__max_depth=RFR_max_depth,
                             clf__min_samples_split=RFR_min_samples_split,
                             clf__min_samples_leaf=RFR_min_samples_leaf,
                             clf__max_features='sqrt',
                             clf__bootstrap=False,
                             clf__n_jobs=-1,
                             clf__tol=1e-04,clf__n_iter=1000,clf__penalty='l2',clf__loss='squared_loss')
            fit = fit_function(model, evaluation, X_train, y_train, X_dev, y_dev, X_test, y_test, params, res_dev_list, res_test_list, params_list)

    res_dev = fit[0][fit[1].index(max(fit[1], key=lambda x:x["pearson"]))]
    res_test = max(fit[1], key=lambda x:x["pearson"])

    return res_dev, res_test


def train_ei_oc(emotion, model, algorithm, evaluation, finetune, baseline, preprocessor=None):
    """
    2. Task EI-oc: Detecting Emotion Intensity (ordinal classification)

    Given:

    a tweet
    an emotion E (anger, fear, joy, or sadness)

    Task: classify the tweet into one of four ordinal classes of intensity of E
    that best represents the mental state of the tweeter:

        0: no E can be inferred
        1: low amount of E can be inferred
        2: moderate amount of E can be inferred
        3: high amount of E can be inferred

    For each language: 4 training sets and 4 test sets: one for each emotion E.

    :param emotion: emotions = ["anger", "fear", "joy", "sadness"]
    :param pretrained:
    :param finetune:
    :param unfreeze:
    :return:
    """

    if preprocessor is None:
        preprocessor = twitter_preprocess()

    model_config = TASK1_EIOC

    X_train, y_train = parse(task='EI-oc', emotion=emotion, dataset="train")
    X_dev, y_dev = parse(task='EI-oc', emotion=emotion, dataset="dev")
    X_test, y_test = parse(task='EI-oc', emotion=emotion, dataset="gold")

    # keep only scores
    y_train = [y[1] for y in y_train]
    y_dev = [y[1] for y in y_dev]
    y_test = [y[1] for y in y_test]

    name = model_config["name"] + "_" + emotion

    X_train = preprocessor("{}_{}".format(name, "train"), X_train)
    X_dev = preprocessor("{}_{}".format(name, "dev"), X_dev)
    X_test = preprocessor("{}_{}".format(name, "test"), X_test)

    params = []
    params_list = []
    res_dev_list = []
    res_test_list = []

    if algorithm == 'LoR':    
        if finetune == 'true':
            for LoR_C in numpy.arange(10,1000,5)/100:
                params = (LoR_C)
                print("Now training with parameters: C: {}".format(LoR_C))
                model.set_params(clf__C=LoR_C,clf__solver='saga',clf__n_jobs=-1)
                fit = fit_function(model, evaluation, X_train, y_train, X_dev, y_dev, X_test, y_test, params, res_dev_list, res_test_list, params_list)              
            print("Best result on gold set: ", max(res_test_list, key=lambda x:x["pearson"]))
            print("Best params: ", params_list[res_test_list.index(max(res_test_list, key=lambda x:x["pearson"]))])
        else:
            if emotion == 'joy':
                if baseline == 'true':
                    LoR_C = 3.1
                else:
                    LoR_C = 3.5
            elif emotion == 'sadness':
                    LoR_C = 1.085
            elif emotion == 'fear':
                LoR_C = 3.5
            elif emotion == 'anger':
                if baseline == 'true':
                    LoR_C = 2.25
                else:
                    LoR_C = 3.8
            params = (LoR_C)
            model.set_params(clf__C=LoR_C,clf__solver='saga',clf__n_jobs=-1)
            fit = fit_function(model, evaluation, X_train, y_train, X_dev, y_dev, X_test, y_test, params, res_dev_list, res_test_list, params_list)            
    elif algorithm == 'SVC':
        if finetune == 'true':
            for SVC_C in numpy.arange(10,50,1)/10:
                for SVC_gamma in numpy.arange(65,500,15)/100:
                    params = (SVC_C,SVC_gamma)
                    print("Now training with parameters: C: {}, Gamma: {}".format(SVC_C,SVC_gamma))
                    model.set_params(clf__C=SVC_C,clf__gamma=SVC_gamma)
                    fit = fit_function(model, evaluation, X_train, y_train, X_dev, y_dev, X_test, y_test, params, res_dev_list, res_test_list, params_list)
            print("Best result on gold set: ", max(fit[1], key=lambda x:x["pearson"]))
            print("Best params: ", fit[2][res_test_list.index(max(res_test_list, key=lambda x:x["pearson"]))])
        else:
            if emotion == 'joy':
                if baseline == 'true':
                    SVC_C = 3.1
                    SVC_gamma = 0.95
                else:
                    SVC_C = 2.7
                    SVC_gamma = 3.35
            elif emotion == 'sadness':
                if baseline == 'true':
                    SVC_C = 2.5
                    SVC_gamma = 1.25
                else:
                    SVC_C = 2.2
                    SVC_gamma = 2.6
            elif emotion == 'fear':
                if baseline == 'true':
                    SVC_C = 2.1
                    SVC_gamma = 0.65 
                else:
                    SVC_C = 4.9
                    SVC_gamma = 4.4 
            elif emotion == 'anger':
                if baseline == 'true':
                    SVC_C = 1.8
                    SVC_gamma = 1.7 
                else:
                    SVC_C = 2.6
                    SVC_gamma = 4.4 
            params = (SVC_C,SVC_gamma)
            model.set_params(clf__C=SVC_C,clf__gamma=SVC_gamma)
            fit = fit_function(model, evaluation, X_train, y_train, X_dev, y_dev, X_test, y_test, params, res_dev_list, res_test_list, params_list)
    res_dev = fit[0][fit[1].index(max(fit[1], key=lambda x:x["pearson"]))]
    res_test = max(fit[1], key=lambda x:x["pearson"])

    return res_dev, res_test


def train_v_oc(model, algorithm, evaluation, finetune, baseline, preprocessor=None):
    """
    4. Task V-oc: Detecting Valence (ordinal classification)
    -- This is the traditional Sentiment Analysis Task

    Given:
     - a tweet

    Task: classify the tweet into one of seven ordinal classes,
    corresponding to various levels of positive and negative sentiment
    intensity, that best represents the mental state of the tweeter:

        3: very positive mental state can be inferred
        2: moderately positive mental state can be inferred
        1: slightly positive mental state can be inferred
        0: neutral or mixed mental state can be inferred
        -1: slightly negative mental state can be inferred
        -2: moderately negative mental state can be inferred
        -3: very negative mental state can be inferred

    For each language: 1 training set, 1 test set.

    :param pretrained:
    :param finetune:
    :param unfreeze:
    :return:
    """
    model_config = TASK1_VOC

    # load the dataset and split it in train and val sets
    X_train, y_train = parse(task='V-oc', dataset="train")
    X_dev, y_dev = parse(task='V-oc', dataset="dev")
    X_test, y_test = parse(task='V-oc', dataset="gold")

    # keep only scores
    y_train = [str(y[1]) for y in y_train]
    y_dev = [str(y[1]) for y in y_dev]
    y_test = [str(y[1]) for y in y_test]

    name = model_config["name"]

    X_train = preprocessor("{}_{}".format(name, "train"), X_train)
    X_dev = preprocessor("{}_{}".format(name, "dev"), X_dev)
    X_test = preprocessor("{}_{}".format(name, "test"), X_test)

    params = []
    params_list = []
    res_dev_list = []
    res_test_list = []
    
    if algorithm == 'LoR':
        if finetune == 'true':
            for LoR_C in numpy.arange(1,100,1)/10:
                params = (LoR_C)
                print("Now training with parameters: C: {}".format(LoR_C))
                model.set_params(clf__C=LoR_C,clf__class_weight='balanced',clf__n_jobs=-1)
                fit = fit_function(model, evaluation, X_train, y_train, X_dev, y_dev, X_test, y_test, params, res_dev_list, res_test_list, params_list)               
            print("Best result on gold set: ", max(res_test_list, key=lambda x:x["pearson"]))
            print("Best params: ", params_list[res_test_list.index(max(res_test_list, key=lambda x:x["pearson"]))])
        else:
            LoR_C = 0.75
            params = (LoR_C)
            model.set_params(clf__C=LoR_C,clf__class_weight='balanced',clf__n_jobs=-1)
            fit = fit_function(model, evaluation, X_train, y_train, X_dev, y_dev, X_test, y_test, params, res_dev_list, res_test_list, params_list)
          
    elif algorithm == 'SVC':
        if finetune == 'true':
            for SVC_C in numpy.arange(10,50,1)/10:
                for SVC_gamma in numpy.arange(65,500,15)/100:
                    params = (SVC_C,SVC_gamma)
                    print("Now training with parameters: C: {}, Gamma: {}".format(SVC_C,SVC_gamma))
                    model.set_params(clf__C=SVC_C,clf__gamma=SVC_gamma)
                    fit = fit_function(model, evaluation, X_train, y_train, X_dev, y_dev, X_test, y_test, params, res_dev_list, res_test_list, params_list)
            print("Best result on gold set: ", max(fit[1], key=lambda x:x["pearson"]))
            print("Best params: ", fit[2][res_test_list.index(max(res_test_list, key=lambda x:x["pearson"]))])
        else:
            SVC_C = 2.5
            SVC_gamma = 0.65
            params = (SVC_C,SVC_gamma)
            model.set_params(clf__C=SVC_C,clf__gamma=SVC_gamma)
            fit = fit_function(model, evaluation, X_train, y_train, X_dev, y_dev, X_test, y_test, params, res_dev_list, res_test_list, params_list)
    
    res_dev = fit[0][fit[1].index(max(fit[1], key=lambda x:x["pearson"]))]
    res_test = max(fit[1], key=lambda x:x["pearson"])

    return res_dev, res_test
