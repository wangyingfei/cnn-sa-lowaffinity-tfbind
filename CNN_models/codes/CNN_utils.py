# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 17:33:30 2019

@author: bxin
"""
import os
import keras
import numpy as np
from keras import backend as K
from keras.optimizers import Adam
from keras.layers.core import Dropout
import keras.regularizers as kr 
import matplotlib.pyplot as plt
import csv
from collections import OrderedDict 
from sklearn.utils.class_weight import compute_class_weight


def parseargs(dataname, args):
    args.add_argument("--tfs", type=str, help="A comma-seprated list of TFs to be ran.")
    args.add_argument("--lr_file", type=str, help="A TXT file, each line presents learning rate in the order of lr_conv1, lr_conv2, lr_backprop.")
    args = args.parse_args()
       
    args.tfs = args.tfs.split(",")
    
    args.outdir = "../out/" + dataname  # ../out/SELEX
    
    return args

def CNN(lr1, lr2, input_len, input_channel=4, nb_filter=100, dense_dim=512, output_len=1, \
            filter_len=11, pool_len=2, dropout=0.2, nDense=1, nConv=1):
    print("Using canonical CNN methods.")
    model = keras.models.Sequential()
    
    #Convolutional layers
    w_reg = kr.WeightRegularizer(l1=lr1, l2=lr2)
    #first conv layer
    model.add(keras.layers.convolutional.Convolution1D(input_shape=(input_len,input_channel),
                                                       nb_filter=nb_filter,
                                                       activation='relu',
                                                       border_mode='same',
                                                       init='glorot_uniform',
                                                       W_regularizer=w_reg,
                                                       filter_length=filter_len))
    model.add(keras.layers.core.Activation("relu"))                                                   
    model.add(keras.layers.pooling.MaxPooling1D(pool_length=pool_len))
    if nConv > 1:
        for i in range(nConv-1):
            model.add(keras.layers.convolutional.Convolution1D(nb_filter=nb_filter,
                                                       border_mode='same',
                                                       filter_length=filter_len))
            model.add(keras.layers.pooling.MaxPooling1D(pool_length=pool_len))
    
    model.add(keras.layers.core.Flatten())
    
    for i in range(nDense):    
        model.add(keras.layers.core.Dense(output_dim=dense_dim, activation='linear',
                                    init='glorot_uniform', bias=True))
        model.add(keras.layers.core.Activation("relu"))
        model.add(Dropout(dropout))
    
    model.add(keras.layers.core.Dense(output_dim=output_len, activation='sigmoid',
                                init='glorot_uniform', bias=True))
    return model

def RC_CNN(lr1, lr2, input_len, input_channel=4, nb_filter=32, dense_dim=512, output_len=1, \
            filter_len=11, pool_len=2, dropout=0.2, nDense=1, nConv=1):
    print("Using revcomp weight sharing methods.")
    model = keras.models.Sequential()
    
    #first layer
    w_reg = kr.WeightRegularizer(l1=lr1, l2=lr2)
    model.add(keras.layers.convolutional.RevCompConv1D(input_shape=(input_len,input_channel),
                                                       nb_filter=nb_filter,
                                                       activation='relu',
                                                       border_mode='same',
                                                       init='glorot_uniform',
                                                       W_regularizer=w_reg,
                                                       filter_length=filter_len))                                                 
    model.add(keras.layers.normalization.RevCompConv1DBatchNorm())
    model.add(keras.layers.core.Activation("relu"))
                                   
    #second layer
    if nConv > 1:
        for i in range(nConv-1):
            model.add(keras.layers.convolutional.RevCompConv1D(nb_filter=nb_filter,
                                                               border_mode='same',
                                                               filter_length=filter_len))
            model.add(keras.layers.normalization.RevCompConv1DBatchNorm())
            model.add(keras.layers.core.Activation("relu"))
    
    #weighted sum layer
    model.add(keras.layers.pooling.MaxPooling1D(pool_length=pool_len))
    model.add(keras.layers.convolutional.WeightedSum1D(symmetric=False,
                                                       input_is_revcomp_conv=True,
                                                       bias=False,
                                                       init="fanintimesfanouttimestwo"))
                                                       
    model.add(keras.layers.core.DenseAfterRevcompWeightedSum(output_dim=dense_dim, activation='linear',
                                init='glorot_uniform', bias=True))
                                
    model.add(keras.layers.core.Activation("relu"))
    
    if nDense > 1:
        for i in range(nDense-1):
            model.add(keras.layers.core.Dense(output_dim=dense_dim, activation='linear',
                                        init='glorot_uniform', bias=True))
            model.add(keras.layers.core.Activation("relu"))
            model.add(Dropout(dropout))
    
    model.add(keras.layers.core.Dense(output_dim=output_len, activation='sigmoid',
                                init='glorot_uniform', bias=True))
    return model
    
def seq_and_HM_CNN(lr1, lr2, input_len_seq,input_channel_seq, input_len_hm, input_channel_hm, nb_filter_seq=[20], \
            nb_filter_hm=[20], dense_dim=[128], output_len=1, filter_len_seq=6,filter_len_hm=6, pool_len=2, dropout=0.2, \
            nDense=1,nConv=1, add_seq=True, add_hm=True):
    print("Using seq_and_HM_CNN methods.")
    if add_seq:
        seq_model = keras.models.Sequential()
        w_reg_seq = kr.WeightRegularizer(l1=lr1, l2=lr2)
        seq_model.add(keras.layers.convolutional.Convolution1D(input_shape=(input_len_seq,input_channel_seq),
                                                       nb_filter=nb_filter_seq[0],
                                                       activation='relu',
                                                       border_mode='same',
                                                       init='glorot_uniform',
                                                       W_regularizer=w_reg_seq,
                                                       filter_length=filter_len_seq))
        seq_model.add(keras.layers.core.Activation("relu"))                                                   
        seq_model.add(keras.layers.pooling.MaxPooling1D(pool_length=pool_len))
        if len(nb_filter_seq) > 1:
            for nb_filter in nb_filter_seq[1:]:
                seq_model.add(keras.layers.convolutional.Convolution1D(nb_filter=nb_filter,
                                                       border_mode='same',
                                                       filter_length=filter_len_seq))
                seq_model.add(keras.layers.pooling.MaxPooling1D(pool_length=pool_len))  
        seq_model.add(keras.layers.core.Flatten())
    
    if add_hm == 1:       
        hm_model = keras.models.Sequential()
        w_reg_hm = kr.WeightRegularizer(l1=lr1, l2=lr2)
        hm_model.add(keras.layers.convolutional.Convolution1D(input_shape=(input_len_hm,input_channel_hm),
                                                       nb_filter=nb_filter_hm[0],
                                                       activation='relu',
                                                       border_mode='same',
                                                       init='glorot_uniform',
                                                       W_regularizer=w_reg_hm,
                                                       filter_length=filter_len_hm))
        hm_model.add(keras.layers.core.Activation("relu")) 
        hm_model.add(keras.layers.pooling.MaxPooling1D(pool_length=pool_len))
        if len(nb_filter_hm) > 1:
            for nb_filter in nb_filter_hm[1:]:
                hm_model.add(keras.layers.convolutional.Convolution1D(nb_filter=nb_filter,
                                                       border_mode='same',
                                                       filter_length=filter_len_hm))
                hm_model.add(keras.layers.pooling.MaxPooling1D(pool_length=pool_len))   
        hm_model.add(keras.layers.core.Flatten())
    
    if add_seq and add_hm:        
        merged = keras.models.Merge([seq_model, hm_model], mode= 'concat')
        model = keras.models.Sequential()
        model.add(merged)
    elif add_seq:
        model = seq_model
    else:
        model = hm_model
        
    for dd in dense_dim:    
        model.add(keras.layers.core.Dense(output_dim=dd, activation='linear',
                                    init='glorot_uniform', bias=True))
        model.add(keras.layers.core.Activation("relu"))
        model.add(Dropout(dropout))    
    model.add(keras.layers.core.Dense(output_dim=2, activation='softmax',
                                init='glorot_uniform', bias=True))  

def r_squared(obs, pred):
    # y is observed values, z is predicted value    
    mse = np.mean((obs - pred)**2)
    return 1-mse/np.mean((obs-np.mean(obs))**2)
    
def train_SELEX(outdir, onehot, seqs, y, lr_file, train_mode="canonical", nfold=3, batch_size=128, \
        nb_epoch=100, ntrial=6, num_to_save=1):
    np.random.seed(1)
    K.set_learning_phase(0)
    
    '''load lr_file'''
    params = [line.rstrip().split(',') for line in open(lr_file).readlines()]
    
    '''split the index of train data onehot into nfold'''
    index_nfold = np.array_split(np.arange(len(seqs)), nfold)

    '''run experiment:
        1. for each param, run model on 2 fold and record performance on the 3rd fold, 3 runs in total
        2. remember average performance of 3 runs, as the score of this param
        3. select params with highest num_to_save scores
        4. run this params on all 3 folds ntrial=6 times, save the model, and params, performance on onehot'''
    perfs = np.zeros((len(params), nfold))
    for i, param in enumerate(params):
        lr_conv1, lr_conv2, lr_backprop = np.float(param[0]), np.float(param[1]), np.float(param[2])
        if train_mode in ["canonical", "double", "RCaugmented"]:
            model = CNN(lr_conv1, lr_conv2, len(seqs[0]))
        else:
            model = RC_CNN(lr_conv1, lr_conv2, len(seqs[0]))
        
        opt = Adam(lr=lr_backprop)
        model.compile(optimizer=opt, loss="mse")  
    
        for j in np.arange(nfold):
            print ("Parameter set " + str(i) + ", " + ",".join(param) + ", fold " + str(j) + "\n")
            index_tr = np.delete(np.arange(len(seqs)), index_nfold[j])
            onehot_tr, y_tr = onehot[index_tr,], y[index_tr,]
            onehot_val, y_val = onehot[index_nfold[j],], y[index_nfold[j],]
            
            model.fit(onehot_tr, y_tr, batch_size = batch_size, nb_epoch = nb_epoch, verbose = 2)
            y_val_hat = model.predict(onehot_val, batch_size, verbose=2)
            perfs[i,j] = r_squared(y_val, y_val_hat)
    print (perfs)
    print ("Start training all train data...")
    best_params = []
    for i in np.mean(perfs, axis=1).argsort()[-num_to_save:][::-1]: 
        best_params.append(params[i]) 
    outfile = open(outdir + "/train_perf.txt", 'w')
    for k in np.arange(num_to_save):
        lr_conv1, lr_conv2, lr_backprop = np.float(best_params[k][0]), np.float(best_params[k][1]), np.float(best_params[k][2])
        if train_mode in ["canonical", "double", "RCaugmented"]:
            model = CNN(lr_conv1, lr_conv2, len(seqs[0]))
        else:
            model = RC_CNN(lr_conv1, lr_conv2, len(seqs[0]))
        
        opt = Adam(lr=lr_backprop)
        model.compile(optimizer=opt, loss="mse")
        
        train_perfs = 0
        for z in np.arange(ntrial):
            print ("Model " + str(k) + ", " + ",".join(best_params[k]) + ", Trial " + str(z) + "\n")
            model.fit(onehot, y, batch_size = batch_size, nb_epoch = nb_epoch, verbose = 2)
            y_hat = model.predict(onehot, batch_size, verbose=2)
            if r_squared(y, y_hat) > train_perfs:
                train_perfs = r_squared(y, y_hat)
                outmodel = open(outdir + "model_"+ ",".join(best_params[k]) + ".json", 'w')
                outmodel.write(model.to_json())
                outmodel.close()
                model.save_weights(outdir + "model_" + ",".join(best_params[k]) + ".h5")        
        outfile.write(",".join(best_params[k]) + "\t" + str(train_perfs) + "\n")
        print (",".join(best_params[k]) + "\t" + str(train_perfs) + "\n")
    outfile.close()

def simple_train(outdir, onehot, seqs, y, lr_file, train_mode="RCmodel", batch_size=128, \
        nb_epoch=100, ntrial=6):
    '''this is mainly for generating overfitting data for figure 2B. So that performance for all parameters will be reported. '''
    np.random.seed(1)
    K.set_learning_phase(0)
    
    '''load lr_file'''
    params = [line.rstrip().split(',') for line in open(lr_file).readlines()]
    save_params = [line.rstrip().split('\t')[0] for line in open(outdir+"/train_perf.txt").readlines()]
    outfile = open(outdir + "/simple_train_perf.txt", 'w')
    for i, param in enumerate(params):
        if ",".join(param) in save_params:
            continue
        lr_conv1, lr_conv2, lr_backprop = np.float(param[0]), np.float(param[1]), np.float(param[2])
        model = RC_CNN(lr_conv1, lr_conv2, len(seqs[0]))
        
        opt = Adam(lr=lr_backprop)
        model.compile(optimizer=opt, loss="mse") 
        
        train_perfs = 0
        for z in np.arange(ntrial):
            print ("Model " + str(i) + ", " + ",".join(param) + ", Trial " + str(z) + "\n")
            model.fit(onehot, y, batch_size = batch_size, nb_epoch = nb_epoch, verbose = 2)
            y_hat = model.predict(onehot, batch_size, verbose=2)
            if r_squared(y, y_hat) > train_perfs:
                train_perfs = r_squared(y, y_hat)
                outmodel = open(outdir + "model_"+ ",".join(param) + ".json", 'w')
                outmodel.write(model.to_json())
                outmodel.close()
                model.save_weights(outdir + "model_" + ",".join(param) + ".h5")        
        outfile.write(",".join(param) + "\t" + str(train_perfs) + "\n")
        print (",".join(param) + "\t" + str(train_perfs) + "\n")
    outfile.close()       

def load_model(json_file, weight_file):
        from keras.models import model_from_json
        json_read = open(json_file, 'r')
        loaded_model_json = json_read.read()
        json_read.close()
        model = model_from_json(loaded_model_json)
        # load weights into new model
        model.load_weights(weight_file)
        return model
       
def predict_SELEX(outdir, pred_file, param_list, onehot_test, seqs_test, y_test, batch_size = 128, fwd_vs_rev=True, plot=False):
    '''default is to look at train_perf and find the best param setting to predict.
    If param is set, use the param to predict, params is the form of a str, using comma seperate lr_conv1, lr_conv2, lr_backprop'''
    
    '''param_list has several lines, the first column of each line represent the parameter setting, \
    a str in lr_conv1, lr_conv2, lr_backprop format. 
    Param_list could have multiple columns, like in train_perf.txt'''    
    params = [line.rstrip().split('\t')[0].split(',') for line in open(param_list).readlines()]
    if fwd_vs_rev == False:    
        outfile = open(outdir + "/test_perf.txt", 'w')

    for i, param in enumerate(params):
        lr_backprop = np.float(param[2])
        print ("Start loading model..." + "for parameter " + str(i) + ", "+ ",".join(param) + "\n")
        model = load_model(outdir + "/model_"+ ",".join(param) + ".json", outdir + "/model_" + ",".join(param) + ".h5")
  
        opt = Adam(lr=lr_backprop)
        model.compile(optimizer=opt, loss="mse")
        
        if fwd_vs_rev == False:        
            y_test_hat = model.predict(onehot_test, batch_size, verbose=2)
            outfile.write(",".join(param) + "\t" + str(r_squared(y_test, y_test_hat)) + "\n")
            print ("Test performance: " + str(r_squared(y_test, y_test_hat)) + "\n")
        else:
            outfile_prefix = outdir + "/" + pred_file.split('/')[-1][:-3] + "_" + ",".join(param)
            fwd_predict = model.predict(onehot_test)
            rev_predict = model.predict(onehot_test[:, ::-1, ::-1])
            seqs_test = seqs_test.reshape(len(seqs_test),1)
            np.savetxt(outfile_prefix + ".txt", np.concatenate((seqs_test, fwd_predict, rev_predict),axis=1), delimiter='\t', fmt="%s\t%1.8f\t%1.8f")
            if plot:
                plt.figure()
                plt.scatter(fwd_predict, rev_predict)
                plt.savefig(outfile_prefix + ".png")
    if fwd_vs_rev == False:
        outfile.close()   

def simple_predict(outdir, param_list, onehot_test, y_test, batch_size = 128, fwd_vs_rev=False):
    '''similar to simple_train, get the performance of each model on test data set.'''    
    params = [line.rstrip().split('\t')[0].split(',') for line in open(param_list).readlines()]

    save_params = [line.rstrip().split('\t')[0] for line in open(outdir+"/train_perf.txt").readlines()]
    outfile = open(outdir + "/simple_test_perf.txt", 'w')
    for i, param in enumerate(params):
        if ",".join(param) in save_params:
            continue
        lr_backprop = np.float(param[2])
    
        print ("Start loading model..." + "for parameter " + str(i) + ", "+ ",".join(param) + "\n")
        model = load_model(outdir + "/model_"+ ",".join(param) + ".json", outdir + "/model_" + ",".join(param) + ".h5")
  
        opt = Adam(lr=lr_backprop)
        model.compile(optimizer=opt, loss="mse")
    
        y_test_hat = model.predict(onehot_test, batch_size, verbose=2)
        outfile.write(",".join(param) + "\t" + str(r_squared(y_test, y_test_hat)) + "\n")
        print ("Test performance: " + str(r_squared(y_test, y_test_hat)) + "\n")

        if fwd_vs_rev:
            fwd_predict = model.predict(onehot_test)
            rev_predict = model.predict(onehot_test[:, ::-1, ::-1])
            np.savetxt(outdir + "/fwd_vs_rev_" + ",".join(param) + ".txt", np.concatenate((fwd_predict, rev_predict),axis=1), delimiter='\t')
            plt.figure()
            plt.scatter(fwd_predict, rev_predict)
            plt.savefig(outdir +  "/fwd_vs_rev_" + ",".join(param) + ".png")
    outfile.close() 
    
def train_encode(outdir, onehot, seqs, hm, y, lr_file, tf_len, nfold=3, batch_size=128, \
        nb_epoch=100, ntrial=6, num_to_save=1, add_seq=True, add_hm=True):
    np.random.seed(1)
    K.set_learning_phase(0)
    
    '''load lr_file'''
    params = [line.rstrip().split(',') for line in open(lr_file).readlines()]
    
    '''split the index of train data onehot into nfold'''
    index_nfold = np.array_split(np.arange(len(seqs)), nfold)

    '''run experiment:
        1. for each param, run model on 2 fold and record performance on the 3rd fold, 3 runs in total
        2. remember average performance of 3 runs, as the score of this param
        3. select params with highest num_to_save scores
        4. run this params on all 3 folds ntrial=6 times, save the model, and params, performance on onehot'''
    perfs = np.zeros((len(params), nfold))
    _, input_len_seq, input_channel_seq = onehot.shape
    _, input_len_hm, input_channel_hm = hm.shape

    for i, param in enumerate(params):
        lr_conv1, lr_conv2, lr_backprop = np.float(param[0]), np.float(param[1]), np.float(param[2])
        model = seq_and_HM_CNN(lr_conv1, lr_conv2, input_len_seq, input_channel_seq, input_len_hm, input_channel_hm,\
        filter_len_seq = int(tf_len) +2, filter_len_hm = int(tf_len)+2, add_seq = add_seq, add_hm = add_hm)

        opt = Adam(lr=lr_backprop)
        model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=['accuracy', 'f1score', 'precision', 'recall'])   #for regression models

        for j in np.arange(nfold):
            print ("Parameter set " + str(i) + ", " + ",".join(param) + ", fold " + str(j) + "\n")
            index_tr = np.delete(np.arange(len(seqs)), index_nfold[j])
            onehot_tr, hm_tr, y_tr = onehot[index_tr,], hm[index_tr,], y[index_tr,]
            onehot_val, hm_val, y_val = onehot[index_nfold[j],], hm[index_nfold[j],], y[index_nfold[j],]
            
            labels = keras.utils.np_utils.to_categorical(y_tr, 2)
            y_integers = np.argmax(labels, axis=1)
            class_weights = compute_class_weight('balanced', np.unique(y_integers), y_integers)
            d_class_weight = dict(enumerate(class_weights))
            
            if add_hm and add_seq:
                model.fit([onehot_tr, hm_tr], labels, batch_size = batch_size, nb_epoch = nb_epoch, class_weight=d_class_weight, verbose = 2)
            elif add_seq:
                model.fit(onehot_tr, labels, batch_size = batch_size, nb_epoch = nb_epoch, class_weight=d_class_weight, verbose = 2)
            
            val_labels = keras.utils.np_utils.to_categorical(y_val,2)

            if add_hm and add_seq: 
                score = model.evaluate([onehot_val, hm_val], val_labels, batch_size)
            elif add_seq:
                score = model.evaluate(onehot_val, val_labels, batch_size)           
            perfs[i,j] = score
    print (perfs)
    print ("Start training all train data...")
    best_params = []
    for i in np.mean(perfs, axis=1).argsort()[-num_to_save:][::-1]: 
        best_params.append(params[i]) 
    outfile = open(outdir + "/train_perf.txt", 'w')
    for k in np.arange(num_to_save):
        lr_conv1, lr_conv2, lr_backprop = np.float(best_params[k][0]), np.float(best_params[k][1]), np.float(best_params[k][2])
        model = seq_and_HM_CNN(lr_conv1, lr_conv2, input_len_seq, input_channel_seq, input_len_hm, input_channel_hm,\
        filter_len_seq = int(tf_len) +2, filter_len_hm = int(tf_len)+2, add_seq = add_seq, add_hm = add_hm)
        
        opt = Adam(lr=lr_backprop)
        model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=['accuracy', 'f1score', 'precision', 'recall'])   #for regression models

        train_perfs = 0
        for z in np.arange(ntrial):
            print ("Model " + str(k) + ", " + ",".join(best_params[k]) + ", Trial " + str(z) + "\n")
            labels = keras.utils.np_utils.to_categorical(y, 2)
            y_integers = np.argmax(labels, axis=1)
            class_weights = compute_class_weight('balanced', np.unique(y_integers), y_integers)
            d_class_weight = dict(enumerate(class_weights))
            
            if add_hm and add_seq:
                model.fit([onehot, hm], labels, batch_size = batch_size, nb_epoch = nb_epoch, class_weight=d_class_weight, verbose = 2)
            elif add_seq:
                model.fit(onehot, labels, batch_size = batch_size, nb_epoch = nb_epoch, class_weight=d_class_weight, verbose = 2)
            
            if add_hm and add_seq: 
                score = model.evaluate([onehot, hm], labels, batch_size)
            elif add_seq:
                score = model.evaluate(onehot, labels, batch_size)
                
            if score > train_perfs:
                train_perfs = score
                outmodel = open(outdir + "model_"+ ",".join(best_params[k]) + ".json", 'w')
                outmodel.write(model.to_json())
                outmodel.close()
                model.save_weights(outdir + "model_" + ",".join(best_params[k]) + ".h5")        
        outfile.write(",".join(best_params[k]) + "\t" + str(train_perfs) + "\n")
        print (",".join(best_params[k]) + "\t" + str(train_perfs) + "\n")
    outfile.close()
    
def predict_encode(outdir, param_list, onehot_test, seqs_test, hm_test, y_test, batch_size = 128, add_seq=True, add_hm=True):
    '''default is to look at train_perf and find the best param setting to predict.
    If param is set, use the param to predict, params is the form of a str, using comma seperate lr_conv1, lr_conv2, lr_backprop'''
    
    '''param_list has several lines, the first column of each line represent the parameter setting, \
    a str in lr_conv1, lr_conv2, lr_backprop format. 
    Param_list could have multiple columns, like in train_perf.txt'''    
    params = [line.rstrip().split('\t')[0].split(',') for line in open(param_list).readlines()]
    outfile = open(outdir + "/test_perf.txt", 'w')

    for i, param in enumerate(params):
        lr_backprop = np.float(param[2])
        print ("Start loading model..." + "for parameter " + str(i) + ", "+ ",".join(param) + "\n")
        model = load_model(outdir + "/model_"+ ",".join(param) + ".json", outdir + "/model_" + ",".join(param) + ".h5")
  
        opt = Adam(lr=lr_backprop)
        model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=['accuracy', 'f1score', 'precision', 'recall'])   #for regression models

        test_labels = keras.utils.np_utils.to_categorical(y_test,2)

        if add_hm and add_seq: 
            score = model.evaluate([onehot_test, hm_test], test_labels, batch_size)
        elif add_seq:
            score = model.evaluate(onehot_test, test_labels, batch_size)
        outfile.write(",".join(param) + "\t" + str(score) + "\n")
        print ("Test performance: " + str(score) + "\n")
    outfile.close()
    
def read_pred_y(filename):
    with open(filename) as inf:
        reader = csv.reader(inf, delimiter="\t")
        second = zip(*reader)[1]   #seocnd col is the predicted y of forward strand
        return np.array([map(float, second)]) 
        
def deltadeltadeltaG(outdir, param_list, mut_str):
    '''mut_str is one of mut1, mut2, mut3, site1, site2, ....'''
    params = [line.rstrip().split('\t')[0].split(',') for line in open(param_list).readlines()]
    for i, param in enumerate(params):
        outfile = outdir + "/" + mut_str + "_deltadeltadeltaG.txt"
        wt_file = outdir + "long_low_affinity_WT_short_seqs_" + ",".join(param) + ".txt"
        mut_file = outdir + "long_low_affinity_" + mut_str + "_short_seqs_" + ",".join(param) + ".txt"
    
        wt_pred_y = read_pred_y(wt_file)
        mut_pred_y = read_pred_y(mut_file)
        '''calculate the deltadelta(G) of the mut seq at every position.'''
        result = -np.log(np.divide(mut_pred_y, wt_pred_y))
        input_len = 73  # the length of the whole long sequence n. n-k+1 = onehot.shape[0]
        result.resize((input_len,), refcheck=False) # when making long seq to short seqs, num of short seqs is n-k+1, this function tries to output n numbers, rather than n-k+1.
        np.savetxt(outfile, result)
        
def visualize(filename_prefix, matrix, width, title=''):
    '''filename_prefix should be an absolute path of a file. '''  
    '''for short seq, height = 250'''
    '''for long seq, like svb enhancer, height = '''  
    width = 20*matrix.shape[0]+width
    cmd = 'seq2logo -f ' + filename_prefix + '.txt -o ' +  filename_prefix + '.png -I 5 --colors \'FF0000:T,0000FF:C,FFA500:G,32CD32:A\'' \
    + ' -p ' + str(width) + 'x150 -u \'' + title + '\' -H \'xaxis,fineprint,ends\''
    os.system(cmd)
    
def GradientTimesInput(model, onehot):
    outputTensor = model.output
    inputTensor = model.input
    response = outputTensor #tf.log(outputTensor/(1-outputTensor))  #K.mean(outputTensor[:,0])#
    gradients = K.gradients(response, inputTensor)
    tf_gradients = K.function([inputTensor], gradients)
    evaluate_gradients = tf_gradients([onehot])      #contains gradients for every sample
#    idx = options.idx if options.idx>0 else np.argmax(y_tr)
    visualizeD = np.multiply(evaluate_gradients[0][0], onehot[0,:,:])
    return visualizeD
    
def DeconvNet(model, onehot):
    from vis.utils import utils
    from vis.visualization.saliency import get, ActivationMaximization
    from keras import activations
    
    def _identity(x):
        return x
        
    def _rmsprop(grads, cache=None, decay_rate=0.95):
        if cache is None:
            cache = np.zeros_like(grads)
        cache = decay_rate * cache + (1 - decay_rate) * grads ** 2
        step = -grads / np.sqrt(cache + K.epsilon())
        return step, cache
        
    def _get_seed_input(seed_input):
        input_range=(0, 255)
        desired_shape = (1, ) + K.int_shape(input_tensor)[1:]
        if seed_input is None:
            return utils.random_array(desired_shape, mean=np.mean(input_range),
                                      std=0.05 * (input_range[1] - input_range[0]))
    
        # Add batch dim if needed.
        if len(seed_input.shape) != len(desired_shape):
            seed_input = np.expand_dims(seed_input, 0)
    
        # Only possible if channel idx is out of place.
        if seed_input.shape != desired_shape:
            seed_input = np.moveaxis(seed_input, -1, 1)
        return seed_input.astype(K.floatx())
        
    layer_idx = -1    #target is the output layer        
    backprop_modifier = 'relu'

    model.layers[layer_idx].activation = activations.linear
    model = utils.apply_modifications(model)
        
    modifier_fn = get(backprop_modifier)
    model = modifier_fn(model)
    
    losses = [(ActivationMaximization(model.layers[layer_idx], 0), -1)]   #for regression models, use 0 as 

    input_tensor = model.input
    wrt_tensor = input_tensor
    norm_grads = False
    
    loss_names = []
    loss_functions = []
    overall_loss = None
    for loss, weight in losses:
        # Perf optimization. Don't build loss function with 0 weight.
        if weight != 0:
            loss_fn = weight * loss.build_loss()
            overall_loss = loss_fn if overall_loss is None else overall_loss + loss_fn
            loss_names.append(loss.name)
            loss_functions.append(loss_fn)
    
    # Compute gradient of overall with respect to `wrt` tensor.
    grads = K.gradients(overall_loss, wrt_tensor)[0]
    if norm_grads:
        grads = grads / (K.sqrt(K.mean(K.square(grads))) + K.epsilon())
    
    compute_fn = K.function([input_tensor],
                                 loss_functions + [overall_loss, grads, wrt_tensor])

############################################################################ 
    final_grads = np.zeros((onehot.shape[0], onehot.shape[1], onehot.shape[2]))
                                                         
    seed_input = _get_seed_input(onehot[0])
    input_modifiers = None or []
    grad_modifier = _identity
    
    cache = None
    best_loss = float('inf')
#        best_input = None
    
    temp_grads = None
    wrt_value = None
    
    for i in range(100):
        # Apply modifiers `pre` step
        for modifier in input_modifiers:
            seed_input = modifier.pre(seed_input)
        # 0 learning phase for 'test'
        computed_values = compute_fn([seed_input])
#            temp_losses = computed_values[:len(loss_names)]
#            named_losses = zip(loss_names, temp_losses)
        total_loss, temp_grads, wrt_value = computed_values[len(loss_names):]
        if temp_grads.shape != wrt_value.shape:
            temp_grads = np.reshape(temp_grads, wrt_value.shape)
        # Apply grad modifier, here is identify function. no modification. 
        temp_grads = grad_modifier(temp_grads)
    
        # Gradient descent update.
        # It only makes sense to do this if wrt_tensor is input_tensor. Otherwise shapes wont match for the update.
        if wrt_tensor is input_tensor:
            step, cache = _rmsprop(temp_grads, cache)
            seed_input += step
    
        # Apply modifiers `post` step
        for modifier in reversed(input_modifiers):
            seed_input = modifier.post(seed_input)
    
        if total_loss < best_loss:
            best_loss = total_loss.copy()
#                best_input = seed_input.copy()

#        result=  utils.normalize(temp_grads)[0]
    final_grads[0, :, :] = temp_grads[0]
    visualizeD = np.multiply(final_grads[0], onehot[0,:,:])
    return visualizeD

def calculate_mutation_for_each_window(model, onehot_i):
    l, c = onehot_i.shape
    mut_pred = np.zeros((l,c))
    for i in range(l):
        for j in range(c):
           temp = np.copy(onehot_i)
           temp[i,:] = 0
           temp[i,j] = 1
           mut_pred[i,j] = model.predict(temp.reshape(1,l,c))[0,0]
    # use the method DeepBind uses in supplemental note 10.1
    ref = np.multiply(onehot_i, mut_pred)
    temp_ref = ref[np.nonzero(ref)]
    ref = np.tile(temp_ref, c).reshape(c,l).T
    
    #for every i,j pair, find the largest predicted_y among (ref, mutaiton and 0)
#    temp_emphasize = np.maximum(ref, mut_pred)
#    emphasize = np.maximum(temp_emphasize, np.zeros((l,c)))   
#    return np.multiply(emphasize, (mut_pred-ref))   #the final output size is l X c
    return np.divide(mut_pred, ref)   #so that mutations will have ratio between (0,1)
    
def calculate_mutation_for_long_seq(model, onehot):   
    '''calculate the mutation map for short seqs from a long seq'''
    temp_n, l, c = onehot.shape  #suppose long seqs have n nucleotide, then temp_n=n-l+1
    n = temp_n + l -1    
    result = np.zeros((n,c))
    
    for i in range(temp_n):  #iterate over all subseqs
        result[i:i+l,:] += calculate_mutation_for_each_window(model, onehot[i,:,:])
    
    if n > 2*(l-1):
        denominator_temp = np.append(np.arange(l)[1:], np.repeat(l,(n-2*(l-1))))
        denominator = np.append(denominator_temp, np.flip(np.arange(l),0)[:-1]) 
    else:
        max_cov = n-l+1
        denominator_temp = np.append(np.arange(n-l+1)[1:], np.repeat(max_cov,(n-2*(n-l))))
        denominator = np.append(denominator_temp, np.flip(np.arange(n-l+1),0)[:-1])
        
    if len(denominator) != result.shape[0]:
        print("In calculate_mutation_for_long_seq, dimsional for last step does not match!!!\n")
        
    return np.divide(result, denominator.reshape(n,1))

def pwm_centralize(pwm):
    '''pwm is of size length X 4'''
    row_mean = pwm.mean(axis=1)
    return pwm-row_mean[:, np.newaxis]
    
def ISM(model, onehot):
    '''onehot is a .h5 file contains 1-bp shifted kmers '''
    '''Different from DeconvNet and GradientTimesInput, ISM here was designed to output the mutation map for a long sequence'''
    '''get the mutation map for both forward strand and reverse complement strand'''
    fwd_mutation_map = calculate_mutation_for_long_seq(model, onehot)   #size is len_of_long_seq X 4
#    rev_mutation_map = calculate_mutation_for_long_seq(model, onehot[:, ::-1, ::-1])
    
    ##save mutation maps for mutation logos
    #first get the original long seqs:
    temp_n, l, c = onehot.shape
    n = temp_n + l -1
    long_seq = np.zeros((n,c))
    for i in range(temp_n):
        long_seq[i:i+l,:] += onehot[i,:,:]
    #long_seq is of size len_of_long_seq X 4
    long_seq[long_seq>0] = 1
    
    #generate the importance of every position by adding the negative values, get a (len_of_long_seq, 1) vector
    #temp_fwd_mutation = np.copy(fwd_mutation_map)
    #temp_fwd_mutation[temp_fwd_mutation>0] = 0
    #temp_fwd_mutation_aggreg = -np.sum(temp_fwd_mutation, axis=1).reshape(n,1)
    
    temp_fwd_mutation = np.copy(fwd_mutation_map)
    '''mainly take care of those 0 elements in the matrix'''
    temp_fwd_mutation[temp_fwd_mutation<=0] = 1e-5
    '''take a log first, then sum up those that have >0 result, meaning ref > mut'''
    temp_fwd_mutation = np.log(temp_fwd_mutation) 
    centralized_temp_fwd_mutation = pwm_centralize(temp_fwd_mutation)
    '''visualize only those appear in the sequence, with original height'''
    visualizeD = np.multiply(long_seq, centralized_temp_fwd_mutation)
    return visualizeD

def interpret_CNN(outdir, method, interpret_file, tf, param_list, onehot):
    np.random.seed(1)
    K.set_learning_phase(0)    
    params = [line.rstrip().split('\t')[0].split(',') for line in open(param_list).readlines()]
    
    for i, param in enumerate(params):
        lr_backprop = np.float(param[2])
    
        print ("Start loading model..." + "for parameter " + str(i) + ", "+ ",".join(param) + "\n")
        model = load_model(outdir + "/model_"+ ",".join(param) + ".json", outdir + "/model_" + ",".join(param) + ".h5")
  
        opt = Adam(lr=lr_backprop)
        model.compile(optimizer=opt, loss="mse")

        if method == "GradientTimesInput":
            gradients = GradientTimesInput(model, onehot)
        elif method == "DeconvNet":
            gradients = DeconvNet(model, onehot)
        elif method == "ISM":
            gradients = ISM(model, onehot)
        else:
            raise NameError('Method names not considered in this study!')

        #one example of filename_prefix could be /home/bxin/Documents/low_affinity/multi-module_CNN/out/SELEX_RCmodel/Scr/ISM/
        filename_prefix = outdir + "/" + method + "/" + interpret_file.split('/')[-1][:-3] + "_" + ",".join(param)
        header = np.array(['A', 'C', 'G','T'], dtype='|S32').reshape(1,4)
        np.savetxt(filename_prefix + '.txt', np.vstack((header, gradients)), delimiter = '\t',fmt='%s')
        
        '''Here the script assumes that GradientTimesInput and DeconvNet only take one sequence in .h5,
        whereas ISM can take either one or multiple sequences for long DNA regions. Nevertheness, every time only one 
        .png file is output'''
        visualize(filename_prefix, gradients, 80, title="")

def set_encoding(seq):
    encoding_dictionary =  OrderedDict ( )
    encoding_dictionary['A'] = [1, 0, 0, 0]
    encoding_dictionary['C'] = [0, 1, 0, 0]
    encoding_dictionary['G'] = [0, 0, 1, 0]
    encoding_dictionary['T'] = [0, 0, 0, 1]
    encoding_dictionary['a'] = [1, 0, 0, 0]
    encoding_dictionary['c'] = [0, 1, 0, 0]
    encoding_dictionary['g'] = [0, 0, 1, 0]
    encoding_dictionary['t'] = [0, 0, 0, 1]   
    encoding_dictionary['N'] = [0, 0, 0, 0]
    encoding_dictionary['n'] = [0, 0, 0, 0]
    encoded = []
    for c in seq:
        encoded.extend(encoding_dictionary[c])
    return np.array(encoded, dtype=np.int8).reshape(len(seq),4)
    
def update_counts_padding_N(counts, n_sites, x, a):
    a_max = a.max(axis=1)
    a_max_idx = a.argmax(axis=1)
    
    n, seq_len, channel_num = x.shape
    nb_filter, filter_len, channel_num = counts.shape
    ##questions here: I should make sure that the last 50 filters are same numbers as the first filters, so that only 
    ## the first 50 filters need to be collected. 
    for i in range(0, n):
        for j in range(0, nb_filter):
            idx = a_max_idx[i, j]
            '''if the window with largest output is partially inside the probe, then delete it'''
            if idx+filter_len <= seq_len:
                counts[j] += a_max[i, j]*x[i, idx:idx+filter_len, :]
            else:
                temp_seq = "N"*(filter_len-seq_len+idx)
                counts[j] += a_max[i,j]*np.concatenate((x[i, idx:idx+filter_len, :], set_encoding(temp_seq)))
            n_sites[j] += 1
    return (counts, n_sites)

def revcompl(s):
    rev_s = ''.join([{'M':'g', 'g':'M', 'A':'T','C':'G','G':'C','T':'A', 'N':'N'}[B] for B in s][::-1])
    return rev_s

def update_each_filter_seqs_append_N(counts, all_input_seqs, filter_id, x, a):
   # a_max = a.max(axis=1)
    a_max_idx = a.argmax(axis=1)
    
    n, seq_len, channel_num = x.shape
    nb_filter, filter_len, channel_num = counts.shape
    
    subseqs = []
    for i in range(0,n):
        idx = a_max_idx[i, filter_id]
        if idx+filter_len <= seq_len:
            subseqs.append(all_input_seqs[i][idx:idx+filter_len])
        else:
            subseqs.append(all_input_seqs[i][idx:idx+filter_len] + "N"*(filter_len-seq_len+idx))
    return subseqs  
#    
#def fitler2motif(outdir, param_list, onehot_train, seqs_train, y_train, onehot_test, seqs_test, y_test, \
#         get_meme=True, align_to_one_filter=False, filter_id=4, use_revcomp=True):
def fitler2motif(outdir, param_list, onehot_train, seqs_train, y_train, \
         get_meme=True, align_to_one_filter=False, filter_id=4, use_revcomp=True):
    '''given a model, onehot data, output the PWM each motif scanner provide.'''
    '''in figure 5D, we used CNN (canonical, RCaugmented) models'''
    np.random.seed(1)
    K.set_learning_phase(0)    
    params = [line.rstrip().split('\t')[0].split(',') for line in open(param_list).readlines()]
    
    for i, param in enumerate(params):
        lr_backprop = np.float(param[2])
    
        print ("Start loading model..." + "for parameter " + str(i) + ", "+ ",".join(param) + "\n")
        model = load_model(outdir + "/model_"+ ",".join(param) + ".json", outdir + "/model_" + ",".join(param) + ".h5")
        opt = Adam(lr=lr_backprop)
        model.compile(optimizer=opt, loss="mse")
        
        N, seq_len, channel_num = onehot_train.shape
        _, act_len, nb_filter = model.layers[0].output_shape
        filter_len, _,channel_num, nb_filter  = model.layers[0].get_weights()[0].shape      
        f = K.function([model.layers[0].input], [model.layers[0].output])
        counts = np.zeros((nb_filter, filter_len, channel_num)) + 1e-5
        n_sites = np.zeros(nb_filter)        
        a = f([onehot_train])
#        b = f([onehot_test])
        
        #%%  use weighted frequencies to visualize frequent matrix
        counts, n_sites = update_counts_padding_N(counts, n_sites, onehot_train, a[0])         
        pwm = counts/counts.sum(axis=2).reshape(nb_filter, filter_len, 1)
        
        if get_meme:
            #write all PWMs as MEME format into a file 
            MEME_HEADER = """MEME version 4.4
            ALPHABET= ACGT
            strands: + -
            Background letter frequencies (from web form):
            A 0.25000 C 0.25000 G 0.25000 T 0.25000 
            """
            base_meme = outdir + '/filters_' + ",".join(param) + '.meme'
            outfile = open(base_meme, 'w')        
            outfile.write(MEME_HEADER)
            
            '''write all pwms in a meme format for further clustering'''
            for i in range(0, nb_filter):
                outfile.write('MOTIF FILTER_LEN%s_%s\n\n' % (filter_len, i))
                outfile.write('letter-probability matrix: alength= 4 w= %s nsites= %s E= 1e-6\n' % (filter_len, int(n_sites[i])))    
                for j in range(0, filter_len):
                    outfile.write('%f\t%f\t%f\t%f\n' % tuple(pwm[i, j, :].tolist()))    
                outfile.write('\n')
            outfile.close()
            
        if align_to_one_filter:
            '''filter_id must be specified so that we know which filter we should align with'''
            '''the filter_id with largest information content (calculated by RSAT) will be recommended'''
            ''' both train and test data were used for alignment'''
            '''use_revcomp=True in case that filters are revcomp to consensus sequence'''
            subseqs_train = update_each_filter_seqs_append_N(counts, seqs_train, filter_id, onehot_train, a[0])
#            subseqs_test = update_each_filter_seqs_append_N(counts, seqs_test, filter_id, onehot_test, b[0])
            out = open(outdir + "/aligned_" + ",".join(param) + "_filter" + str(filter_id) + ".txt", "w")
            if use_revcomp:            
                for s,o in zip(subseqs_train, y_train):
                    out.write(revcompl(s) + '\t' + str(o[0]) + '\n')
#                for s,o in zip(subseqs_test, y_test):
#                    out.write(revcompl(s) + '\t' + str(o[0]) + '\n')
                out.close()
            else:
                for s,o in zip(subseqs_train, y_train):
                    out.write(s + '\t' + str(o[0]) + '\n')
#                for s,o in zip(subseqs_test, y_test):
#                    out.write(s + '\t' + str(o[0]) + '\n')
                out.close()
