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
import matplotlib.pyplot as plt
from keras.regularizers import l1

def parseargs(dataname, args):
    args.add_argument("--tfs", type=str, help="A comma-seprated list of TFs to be ran.")
    args.add_argument("--lr_file", type=str, help="A TXT file, each line presents learning rate in the order of lr_conv1, lr_conv2, lr_backprop.")
    args = args.parse_args()
       
    args.tfs = args.tfs.split(",")
    
    args.outdir = "../out/" + dataname  # ../out/SELEX
    
    return args

def CNN(lr1, lr2, input_len, input_channel=4, nb_filter=100, dense_dim=512, output_len=1, \
            filter_len=11, pool_len=2, dropout=0.2, nDense=1, nConv=1):
    print("Without using revcomp weight sharing methods. Deeplift version.")
    model = keras.models.Sequential()
    
    #Convolutional layers
    #w_reg = kr.WeightRegularizer(l1=lr1, l2=lr2)
    #first conv layer
    model.add(keras.layers.convolutional.Conv1D(input_shape=(input_len,input_channel),
                                                       filters=nb_filter,
                                                       activation='relu',
                                                       padding='same',
                                                       kernel_initializer='glorot_uniform',
                                                       kernel_regularizer=l1(lr1),
                                                       bias_regularizer=l1(lr2),
                                                       kernel_size=filter_len))
    #model.add(keras.layers.core.Activation("relu"))                                                   
    model.add(keras.layers.pooling.MaxPooling1D(pool_size=pool_len))
    if nConv > 1:
        for i in range(nConv-1):
            model.add(keras.layers.convolutional.Convolution1D(filters=nb_filter,
                                                       padding='same',
                                                       kernel_size=filter_len))
            model.add(keras.layers.pooling.MaxPooling1D(pool_size=pool_len))
    
    model.add(keras.layers.core.Flatten())
#    model.add(Dropout(dropout))
    
    for i in range(nDense):    
        model.add(keras.layers.core.Dense(units=dense_dim, activation='linear',
                                    kernel_initializer='glorot_uniform', use_bias=True))
        model.add(keras.layers.core.Activation("relu"))
        model.add(Dropout(dropout))
    
    model.add(keras.layers.core.Dense(units=output_len, activation='sigmoid',
                                      kernel_initializer='glorot_uniform', use_bias=True))
    return model


def r_squared(obs, pred):
    # y is observed values, z is predicted value    
    mse = np.mean((obs - pred)**2)
    return 1-mse/np.mean((obs-np.mean(obs))**2)
    

def simple_train(outdir, onehot, seqs, y, lr_file, train_mode="double", batch_size=128, \
        nb_epoch=100, ntrial=6):
    '''this is mainly for generating overfitting data for figure 2B. So that performance for all parameters will be reported. '''
    np.random.seed(1)
    K.set_learning_phase(0)
    
    '''load lr_file'''
    params = [line.rstrip().split(',') for line in open(lr_file).readlines()]
    outfile = open(outdir + "/simple_train_perf.txt", 'w')
    for i, param in enumerate(params):
        lr_conv1, lr_conv2, lr_backprop = np.float(param[0]), np.float(param[1]), np.float(param[2])
        model = CNN(lr_conv1, lr_conv2, len(seqs[0]))
        
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
    
def simple_predict(outdir, param_list, onehot_test, y_test, batch_size = 128, fwd_vs_rev=False):
    '''similar to simple_train, get the performance of each model on test data set.'''    
    params = [line.rstrip().split('\t')[0].split(',') for line in open(param_list).readlines()]

    outfile = open(outdir + "/simple_test_perf.txt", 'w')
    for i, param in enumerate(params):
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

def visualize(filename_prefix, matrix, width, title=''):
    '''filename_prefix should be an absolute path of a file. '''  
    '''for short seq, height = 250'''
    '''for long seq, like svb enhancer, height = '''  
    width = 20*matrix.shape[0]+width
    cmd = '/Users/beibeixin/Documents/Materials/seq2logo-2.1/Seq2Logo.py -f ' + filename_prefix + '.txt -o ' +  filename_prefix + ' -I 5 --colors \'FF0000:T,0000FF:C,FFA500:G,32CD32:A\'' \
    + ' -p ' + str(width) + 'x120 -u \'' + title + '\' -H \'xaxis,fineprint,ends\' --format \'PDF\'' 
    os.system(cmd)
    
def DeepLIFT(json_file, weight_file, onehot):
    from deeplift.layers import NonlinearMxtsMode
    import deeplift.conversion.kerasapi_conversion as kc
    import deeplift.layers
    import deeplift.conversion.kerasapi_conversion
    from collections import OrderedDict    
    import deeplift
    
    method_to_model = OrderedDict()
    for method_name, nonlinear_mxts_mode in [
        #The genomics default = rescale on conv layers, revealcance on fully-connected
        ('rescale_conv_revealcancel_fc', NonlinearMxtsMode.DeepLIFT_GenomicsDefault),
        ('rescale_all_layers', NonlinearMxtsMode.Rescale),
        ('revealcancel_all_layers', NonlinearMxtsMode.RevealCancel),
        ('grad_times_inp', NonlinearMxtsMode.Gradient),
        ('guided_backprop', NonlinearMxtsMode.GuidedBackprop)]:
        method_to_model[method_name] = kc.convert_model_from_saved_files(
            h5_file=weight_file,
            json_file=json_file,
            nonlinear_mxts_mode=nonlinear_mxts_mode)
    
    print("Compiling scoring functions")
    method_to_scoring_func = OrderedDict()
    for method,model in method_to_model.items():
        print("Compiling scoring function for: "+method)
        method_to_scoring_func[method] = model.get_target_contribs_func(find_scores_layer_idx=0,
                                                                        target_layer_idx=-2)
    
    #To get a function that just gives the gradients, we use the multipliers of the Gradient model
    gradient_func = method_to_model['grad_times_inp'].get_target_multipliers_func(find_scores_layer_idx=0,
                                                                                  target_layer_idx=-2)
                                                                                  
    print("Compiling integrated gradients scoring functions")
    integrated_gradients10_func = deeplift.util.get_integrated_gradients_function(
        gradient_computation_function = gradient_func,
        num_intervals=10)
    method_to_scoring_func['integrated_gradients10'] = integrated_gradients10_func
    
    
    background = OrderedDict([('A', 0.3), ('C', 0.2), ('G', 0.2), ('T', 0.3)])
    
    from collections import OrderedDict
    
    method_to_task_to_scores = OrderedDict()
    for method_name, score_func in method_to_scoring_func.items():
        print("on method",method_name)
        method_to_task_to_scores[method_name] = OrderedDict()
        for task_idx in [0]:
            scores = np.array(score_func(
                        task_idx=task_idx,
                        input_data_list=[onehot],
                        input_references_list=[
                         np.array([background['A'],
                                   background['C'],
                                   background['G'],
                                   background['T']])[None,None,:]],
                        batch_size=200,
                        progress_update=None))
            assert scores.shape[2]==4
            scores = np.sum(scores, axis=2)
            method_to_task_to_scores[method_name][task_idx] = scores
    return method_to_task_to_scores
    

def interpret_CNN(outdir, interpret_file, tf, param_list, onehot):   
    params = [line.rstrip().split(',') for line in open(param_list).readlines()]
    
    for i, param in enumerate(params):    
        print ("Start loading model..." + "for parameter " + str(i) + ", "+ ",".join(param) + "\n")
        json_file = outdir + "/model_"+ ",".join(param) + ".json"
        weight_file = outdir + "/model_" + ",".join(param) + ".h5"
        
        method_to_task_to_scores = DeepLIFT(json_file, weight_file, onehot)
  
      
    ####for visualization
        for method_name in [
                            'grad_times_inp',
                            'guided_backprop',
                            'integrated_gradients10',
                            'rescale_all_layers', 
                            'revealcancel_all_layers',
                            'rescale_conv_revealcancel_fc'
                            ]:
            scores = method_to_task_to_scores[method_name][0]
            for idx in np.arange(onehot.shape[0]):
                scores_for_idx = scores[idx]
                original_onehot = onehot[idx]
                scores_for_idx = original_onehot*scores_for_idx[:,None]

                '''Here the script assumes that DeepLIFT and DeconvNet only take one sequence in .h5, otherwise, saved files will all be the same name. '''
                filename_prefix = outdir + "/" + method_name + "_" + interpret_file.split('/')[-1][:-3] + "_" + ",".join(param)
                header = np.array(['A', 'C', 'G','T'], dtype='|S32').reshape(1,4)
                np.savetxt(filename_prefix + '.txt', np.vstack((header, scores_for_idx)), delimiter = '\t',fmt='%s')
        
                visualize(filename_prefix, scores_for_idx, 80, title="")
        
        
        
