## CNN models

These scripts generate the results of SA models in our manuscript. 

## Dependencies

The pipeline requires:

* python 3.10
* pytorch 1.12.1
* seq2logo 2.1 (Thomsen and Neilson, 2012) at http://www.cbs.dtu.dk/biotools/Seq2Logo/

To set up the environment:
```sh
conda env create -f environment.yaml
```

## Tutorial

1. Build sequence-based SA models on SELEX-seq data. Here we take the SA-raw model for Ubx as an example. 

Optional: Data preparation: input data are .h5 format. if starting from text file, with first and second column indicate sequences and response (i.e. relative affinity), use the following command (in CNN_models/) to prepare .h5 data. 
```sh
cd codes
python generateh5files.py TEXTFILE
```
Use TF_transformer.py to train and same models, arguments in the order of $TFname, $ModelType, $SeqLen, $DateMark
```sh
python TF_transformer.py Ubx canonical 14 0219 > result/Ubx_canonical_gridsearch_0219.txt
```
$ModelType "canonical" referrs to the "SA-raw" model, this option can be changed to "RCaugmented" or "double" for other data augmentation strategies, for RCaugmented, the third argument should be 38 to reflect the input sequence length. The last arugment is a date mark for easy model tracking.

Trained models will be saved into a folder named in this format: $TFname_$ModelType_models_gridsearch_$DateMark/, in this folder, each model is named with their model hyperparameter specifications, such as "lr0.001_batch256_nlayer2_nhead8_dmodel128.pth"


2. Make predictions with trained models

Use predict.py to make predictions on the test set
```sh
python predict.py
# change the input model and data directory in the script for other TFs
```
Or use predict_singleSeq.py to predict a specific sequence with more flexibility. Arguments in the order of $TFname, $ModelType, $SeqLen, $DateMark, $Nlayer, $Nhead, $pred_data_name
Here we use predicting WT svb enhancer using the Ubx-raw model as an example:
```sh
python predict_singleSeq.py Ubx canonical 14 0219 2 8 fullLength_slidingWindow_WT
```
* Note that the sequence to be predicted need to be prepared in advanced in the file: /prediction/data/$pred_data_name.txt, containing 4 columns: sequence, 0 (a dummy values), seq_index (for plotting, can use a dummy value), seq_name (for plotting, can use a dummy value)

Predictions will be saved into /prediction/$TFname_$ModelType_$DateMark_$nlayer_$nhead/$pred_data_name.txt


3. Interpret trained SA models with Gradient*input, SHAP, and ISM methods.

Use interpretation_functions.py to generate unit-resolution importance matrix, arguments in the order of $TFname, $ModelType, $SeqLen, $DateMark, $nhead, $nlayer, $learningRate, $batchSize, $sequence_for_interpretaion

For example, get unit-resolution importance matrix of ATGATTTATTACCC (high-affinity BS of Exd-Ubx) of SA-raw of Exd-Ubx.
```sh
python interpretation_functions.py Ubx canonical 14 0219 2 8 0.001 256 AATGATTAATTGCT
```
Unit-resolution importance matrix will be saved into /predictions/$TF_$ModelType_$SeqLen_$DateMark_$nhead_$nlayer_$learningRate_$batchSize/interpre_seq_$sequence_for_interpretaion_interpretationMethodName.txt

The logos are then generated using the Seq2logo tool with the following command:
```sh
cd seq2logo-2.1
./Seq2Logo.py -f prediction/Ubx_canonical_0219_nlayer2_nhead8_0219_nlayer2_nhead8_lr0.001_batch256/interpret_seq_AATGATTAATTGCT_ISM.txt -o AATGATTAATTGCT_ISM -I 5 --colors 'FF0000:T,0000FF:C,FFA500:G,32CD32:A' -H 'xaxis,fineprint,ends'
```

