## CNN models

These scripts generate the results of CNN models and the L2-MLR model in our manuscript. 

## Dependencies

The pipeline requires:

* python 2.7
* DeepLIFT (Shrikumar et al., 2017) version at https://github.com/kundajelab/deeplift/tree/v0.6.6.2-alpha
* seq2logo (Thomsen and Neilson, 2012) at http://www.cbs.dtu.dk/biotools/Seq2Logo/
* matrix-clustering (Castro-Mondragon et al., 2017) at https://rsat01.biologie.ens.fr/rsat/matrix-clustering_form.cgi

## Tutorial

1. Build the L2-regularized multiple linear regression (L2-MLR) model with aligned SELEX-seq data. Take Exd-Scr for an example. 

```sh
mkdir ../data/SELEX_aligned
cd codes_MLR
R --no-restore --no-save --args ../data/SELEX_aligned/Scr/Scr.txt ../data/SELEX_aligned/Scr/encoded 8 feature_list < encode_custom.R 2>&1 1>/dev/null 
R --no-restore --no-save --args Scr ../data/SELEX_aligned/Scr/input 10 feature_list < shuffle_divide.R 2>&1 1>/dev/null
python train.py Scr ../data/SELEX_aligned/Scr/encoded ../data/SELEX_aligned/Scr/input ../data/SELEX_aligned/Scr/output 10 train.R feature_list
python test.py Scr ../data/SELEX_aligned/Scr/encoded ../data/SELEX_aligned/Scr/input ../data/SELEX_aligned/Scr/output 10 predict.R feature_list
python summarize.py Scr ../data/SELEX_aligned/Scr/output summarize.R ../data/SELEX_aligned/Scr/result feature_list
```

2. Build four sequence-based CNN models on SELEX-seq data. Here we took CNN (RC model) for an example. After running the following command, the top 3 hyperparameter settings would be selected and their corresponding CNN models and training/testing performance would be saved. Note that the prior generated hyperparameter list is in lr_file. 

```sh
# Data preparation: input data are .h5 format. if starting from text file, with first and second column indicate sequences and response (i.e. relative affinity), use the following command to prepare .h5 data. 
cd codes
python generateh5files.py TEXTFILE

# model options are canonical/RCaugmented/RCmodel/double.
python train_CNN_SELEX.py RCmodel --steps train,test --tfs Scr,Ubx,Antp,Pb,Dfd,AbdA,AbdB,Lab --lr_file lr_file 
```

3. Interpret well-trained models with Gradient*input, DeconvNet, and ISM. For example, we could use following commands to obtain unit-resolution importance matrix of ATGATTTATTACCC (high-affinity BS of Exd-Ubx) interpreted by CNN (RC model) of Exd-Ubx.
 
```sh
python interpret_CNN_SELEX.py RCmodel ISM --interpret_file ../data/interpret_seq/ATGATTTATTACCC.h5 --tfs Ubx --lr_file ../out/SELEX_RCmodel/Ubx/train_perf.txt
```

Note that the .h5 file in --interpret_file only contains one sequence for interpretation. For ISM method, the sequence to be interpret could be a long sequence, for example, Svb enhancer as prepared in ../data/long_low_affinity_WT_short_seqs.h5. 

The fourth interpretation method DeepLIFT was implemented separately.
```sh
cd codes_deeplift
python train_CNN_SELEX_deeplift.py double --steps train --tfs Ubx --lr_file lr_file
python train_CNN_SELEX_deeplift.py double --steps interpret --tfs Ubx --lr_file lr_file --interpret_file ../data/interpret_seq/ATGATTTATTACCC.h5
```

4. Free energy change of different mutations on the Svb enhancer. 

```sh
for seq in WT mut1 mut2 mut3 mut12 site1 site2 site3A site3B site3C site3D site3E
do
	for tf in Scr Ubx
	do
 		head -n1 ../out/SELEX_RCmodel/$tf/train_perf.txt > ../out/SELEX_RCmodel/$tf/best_perf.txt
 		python train_CNN_SELEX.py RCmodel --steps predict --tfs $tf --pred_file ../data/interpret_seq/long_low_affinity_$seq\_short_seqs.h5 --lr_file ../out/SELEX_RCmodel/$tf/best_perf.txt
 		python train_CNN_SELEX.py RCmodel --steps deltadeltadeltaG --tfs $tf --mut_str $seq 
done
done 
```

5. After training a CNN model, align training data with one certain motif scanner, and create either a position weight matrix (PWM) or YR logo. Here we use Exd-Ubx and CNN (canonical, RCaugmented) model as an example. 

```sh
# First step is to find out which motif scanner has largest information content.
python train_CNN_SELEX.py RCaugmented --steps align --tfs Ubx
cd ../out/SELEX_RCaugmented/Ubx
matrix-clustering -v 1 -max_matrices 300 -matrix ubx filters_0,0.001,0.0003.meme meme -hclust_method average -calc sum -title 'ubx' -metric_build_tree 'Ncor' -lth w 5 -lth cor 0.6 -lth Ncor 0.4 -quick -label_in_tree name -return json,heatmap -o ./matrix-clustering 2> ./matrix-clustering_err.txt

sort -nk7 ./matrix-clustering_tables/pairwise_compa_matrix_descriptions.tab > test

# Second step: set align_to_one_filter=True, and get aligned sequences based on one filter
cd ../../../codes
python train_CNN_SELEX.py RCaugmented --steps align --tfs Ubx
cd ../out/SELEX_RCaugmented/Ubx
sort -nk2 aligned_0,0.001,0.0003_filter3.txt > aligned_0,0.001,0.0003_filter3_sorted.txt

# Third step: YR coding of high-affinity and low-affinity BSS
head -n10000 aligned_0,0.001,0.0003_filter3_sorted.txt | awk '{print $1}' - | sed s/N/n/g - > aligned_0,0.001,0.0003_filter3_low.txt
tail -n10000 aligned_0,0.001,0.0003_filter3_sorted.txt | awk '{print $1}' - | sed s/N/n/g - > aligned_0,0.001,0.0003_filter3_high.txt
python ../../../codes/YRcodes.py aligned_0,0.001,0.0003_filter3_low.txt aligned_0,0.001,0.0003_filter3_low_YRcodes.txt
python ../../../codes/YRcodes.py aligned_0,0.001,0.0003_filter3_high.txt aligned_0,0.001,0.0003_filter3_high_YRcodes.txt

# Fourth step: visualization
We used the WebLogo 3 (Crooks et al., 2004) online website (http://weblogo.threeplusone.com/create.cgi) to plot PWM and YR logos. For PWM logos, we upload a file (aligned_0,0.001,0.0003_filter3_low.txt or aligned_0,0.001,0.0003_filter3_high.txt), and choose Composition as 'D. melanogaster (43%)' as GC content. 
```


## References

Castro-Mondragon J.A. et al. (2017) RSAT matrix-clustering: dynamic exploration and redundancy reduction of transcription factor binding motif collections. Nucleic Acids Res., 45, e119-e119.

Crooks G.E. et al. (2004) WebLogo: a sequence logo generator. Genome Res., 14, 1188-1190.

Shrikumar A. et al. (2017a) Learning important features through propagating activation differences. arXiv preprint arXiv:1704.02685.

Thomsen M.C. and Nielsen M. (2012) Seq2Logo: a method for construction and visualization of amino acid binding motifs and sequence profiles including sequence weighting, pseudo counts and two-sided representation of amino acid enrichment and depletion. Nucleic Acids Res., 40, W281-W287.
