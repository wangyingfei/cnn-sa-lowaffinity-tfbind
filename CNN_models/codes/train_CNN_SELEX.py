# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 17:27:25 2019

@author: bxin
"""

import sys
import os
import argparse 
import numpy as np
import h5py as h5
import matplotlib.pyplot as plt
import seaborn as sns

import CNN_utils as util

datadir = "../data/SELEX_"

def main():
    args = loadargs()
    
    for tf in args.tfs:
        ensure_dir(args.outdir)
        outdir = args.outdir +"/" + tf + "/"
        ensure_dir(outdir)
        if "train" in args.steps:
            print ("------------train:" + tf + "---------------")
            onehot_tr, seqs_tr, y_tr = read_data(datadir + args.mode + "/" + tf + "/" + tf + "_train.h5" )
            
            plt.figure()
            sns_plot = sns.distplot(y_tr)
            fig = sns_plot.get_figure()
            fig.savefig(outdir + "y_distribution.png")
            
            '''num_to_save means the number of models to be saved at the end of training'''
            util.train_SELEX(outdir, onehot_tr, seqs_tr, y_tr, args.lr_file, train_mode=args.mode, nfold=3, num_to_save=3)
        
        if "test" in args.steps:
            print ("------------test:"+ tf + "----------------")
            onehot_test, seqs_test, y_test = read_data(datadir + args.mode + "/" + tf + "/" + tf + "_test.h5" )            
            util.predict_SELEX(outdir, outdir + "/train_perf.txt", onehot_test, y_test)  
            
        if "predict" in args.steps:
            print ("------------predict:"+ tf + "----------------")
            '''in the predict data, y_pred might be artificial, only need to focus on the output fwd_vs_rev_params.txt, 
            the first column in the file'''
            '''any sequence you want to predict relative binding affinity for, should be created as a h5 file first'''
            onehot_pred, seqs_pred, y_pred = read_data(args.pred_file)
            util.predict_SELEX(outdir, args.pred_file, outdir + "/best_perf.txt", onehot_pred, seqs_pred, y_pred)
        
        if "deltadeltadeltaG" in args.steps:
            print ("------------deltadeltadeltaG:"+ tf + "----------------")
            util.deltadeltadeltaG(outdir, outdir + "/best_perf.txt", args.mut_str)
            
        if "align" in args.steps:
            print ("------------align sequences:"+ tf + "----------------")
            onehot_tr, seqs_tr, y_tr = read_data(datadir + args.mode + "/" + tf + "/" + tf + "_train.h5" )
            
            #first get the .meme file for clustering
            util.fitler2motif(outdir, outdir + "/best_perf.txt", onehot_tr, seqs_tr, y_tr)

            #then pick a filter that used to align, here filter_id equals 3
            util.fitler2motif(outdir, outdir + "/best_perf.txt", onehot_tr, seqs_tr, y_tr, \
            get_meme=False, align_to_one_filter=True, filter_id=3, use_revcomp=True)
            
def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
           
def loadargs():
    args = argparse.ArgumentParser(description="Generate the SELEX experiments.")
    args.add_argument("mode", type=str, help="One of canonical/RCaugmented/RCmodel/double.")
    args.add_argument("--steps", type=str, help="A comma-seperated list of (or subset of) train,test,predict. Default is all.")
    args.add_argument("--pred_file", type=str, help="A .h5 file contains predict sequence. Only useful when predict in args.steps.")        
    args.add_argument("--mut_str", type=str, help="A string being one of mut1, mut2, site 2, site 1 etc. Only useful when deltadeltadeltaG in args.steps.")        
    args = util.parseargs("SELEX", args)
    if args.steps == "all":
        args.steps = "train,test,predict,deltadeltadeltaG"        
    args.steps = args.steps.split(",") 
    args.outdir = args.outdir + "_" + args.mode   # ../out/SELEX_RCmodel
    return args
    
def read_data(h5file):
    f = h5.File(h5file, 'r')
    g = f['data']
    data = g['s_x'].value
    seqs = g['sequence'].value
    targets = g['c0_y'].value
    return (data, seqs, targets)
    
if __name__=="__main__":
    main()