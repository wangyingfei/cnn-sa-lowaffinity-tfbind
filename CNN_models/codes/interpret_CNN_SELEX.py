# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 17:27:25 2019

@author: bxin
"""
import os
import argparse 
import h5py as h5
import CNN_utils as util

datadir = "../data/SELEX_"

def main():
    args = loadargs()
    
    for tf in args.tfs:
        print ("------------Intretation method: " + args.method + " for "+ tf + "---------------")
        onehot_tr, seqs_tr, y_tr = read_data(args.interpret_file)
        
        ensure_dir(args.outdir)
        outdir = args.outdir +"/" + tf + "/" 
        ensure_dir(outdir)
        ensure_dir(outdir + args.method + "/")
        
        '''num_to_save means the number of models to be saved at the end of training'''
        util.interpret_CNN(outdir, args.method, args.interpret_file, tf, args.lr_file, onehot_tr)
        
def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
           
def loadargs():
    args = argparse.ArgumentParser(description="Generate the SELEX experiments.")
    args.add_argument("mode", type=str, help="One of canonical/RCaugmented/RCmodel/double.")
    args.add_argument("method", type=str, help="Interpretation method. One of ISM/DeconvNet/GradientTimesInput")
    args.add_argument("--interpret_file", type=str, help="A .h5 input file (absolute path), contains input sequence for interpretation.\
                    Multiple sequences are allowed and each sequence should have same length as input length of the model.")
    args = util.parseargs("SELEX", args)
    args.outdir = args.outdir + "_" + args.mode    # ../out/SELEX_RCmodel
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