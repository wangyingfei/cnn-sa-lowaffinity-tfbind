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


import CNN_utils_deeplift as util

datadir = "../data/SELEX_"

def main():
    args = loadargs()
    
       
    
    for tf in args.tfs:
        if "train" in args.steps:
            print ("------------train:" + tf + "---------------")
            onehot_tr, seqs_tr, y_tr = read_data(datadir + args.mode + "/" + tf + "/" + tf + "_train.h5" )
            
            ensure_dir(args.outdir)
            outdir = args.outdir +"/" + tf + "/"
            ensure_dir(outdir)
            
            '''num_to_save means the number of models to be saved at the end of training'''
            util.simple_train(outdir, onehot_tr, seqs_tr, y_tr, args.lr_file, train_mode=args.mode)
        if "interpret" in args.steps:
            outdir = args.outdir +"/" + tf + "/"
            onehot, seqs, y = read_data(args.interpret_file)
            util.interpret_CNN(outdir, args.interpret_file, tf, args.lr_file, onehot)

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
           
def loadargs():
    args = argparse.ArgumentParser(description="Generate the SELEX experiments.")
    args.add_argument("mode", type=str, help="One of canonical/RCaugmented/RCmodel/double.")
    args.add_argument("--steps", type=str, help="A comma-seperated list of (or subset of) train,test,predict. Default is all.")
    args.add_argument("--interpret_file", type=str, help=".h5 file, only contains one seq.")
    args = util.parseargs("SELEX", args)
    if args.steps == "all":
        args.steps = "train,test,predict"        
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