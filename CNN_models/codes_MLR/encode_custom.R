# some functions ||
encode_1mer <- function(seq, map){
  features <- c()
  n <- nchar(seq)
  seq <- toupper(seq)
  for (i in 1:n)
    features <- c(features, map[substr(seq, i, i),])

  return(features)
}

complDNA <- function(seq){
    tbl <- list()
    tbl[["a"]] <- "t"
    tbl[["c"]] <- "g"
    tbl[["g"]] <- "c"
    tbl[["t"]] <- "a"
    tbl[["A"]] <- "T"
    tbl[["C"]] <- "G"
    tbl[["G"]] <- "C"
    tbl[["T"]] <- "A"
    complseq <- ""
    for( i in nchar(seq) : 1 ){
        complseq <- paste(complseq, tbl[[substr(seq, i, i)]], sep="")
    }
    return(complseq)
}
# ||

# read arguments
args <- commandArgs(trailingOnly = TRUE)
seqFile <- args[1]
outFolder <- args[2]
palinCore <- F
coreStart <- 5  #if keepcore=T, of course, we need features for all seqs
flanklen <- 4
coreLen <- as.numeric(args[3])  #
keepCore <- T
featureList <- args[4]

# some settings about the input file
cc <- c('character', 'numeric')
seqCol <- 1
sigCol <- 2
logarithm <- F
shuffleShape <- F
normalize <- T
interact <- F
threholdLow <- 8
excludeLow <- F
minSampleSize <- 1

# read data in ...
data <- read.table(seqFile, header=F, colClasses=cc)
## get sequences and signals ++
seqs <- toupper(data[, seqCol])
signals <- data[, sigCol]
## ++
# ...

# test if the sample size is big enough to continue |||
if(excludeLow){
  usedDataIdx <- which(signals > threholdLow) # only use data that has signal > threholdLow
  if(length(usedDataIdx) < minSampleSize){
    stop('Sample size too small')
  }
  ## update seqs and signals to store only data above the threshold ---
  seqs <- seqs[usedDataIdx]
  signals <- signals[usedDataIdx]
  ## ---
}else{
  if(length(signals) < minSampleSize){
    stop('Sample size too small')
  }
}
# |||

# stop when there is signal < 1, if logarithm is set to be true ...
if(logarithm){
  if(sum(signals < 1) > 0){
    stop("You chose to logarize, but there were signal < 1 in the data.")
  }
}
# ...

# get identifier --
identifier <- ""
subs <- unlist(strsplit(seqFile, '/'))
identifierWithExt <- subs[length(subs)]
identifierSubs <- unlist(strsplit(identifierWithExt, '[.]'))
n <- length(identifierSubs)
if(n > 1){
  identifier <- paste0(identifierSubs[1:(n-1)], collapse='.')
}else{
  identifier <- identifierWithExt # there was no extension anyway
}
# --

# re-assemble outFolder (to avoid problem with tailing '/') --
subs <- unlist(strsplit(outFolder, '/'))
outFolder <- paste0(subs, collapse='/')

if (!dir.exists(outFolder)){
    dir.create(outFolder)
} 

# --

# encode features
## construct mapping ...
map_1mer <- rbind(diag(4), diag(4), c(0,0,0,0))
row.names(map_1mer) <- c("A", "C", "G", "T", "a", "c", "g", "t","N")

## encode 1mer/2mer/3mer features...
if(!palinCore){
  ### pre-alloc ...
  n <- nchar(seqs[1])
  m <- length(seqs)
  feature_1mer <- matrix(data=0, nrow=m, ncol=4*n)
  ### ...
  for (i in 1 : m){
     feature_1mer[i,] <- encode_1mer(seqs[i], map_1mer)    
    # #### --- prepare fasta file
    # cat(paste0('>', i), file=tmpFaFile, append=T, fill=T)
    # cat(seqs[i], file=tmpFaFile, append=T, fill=T)
    # #### ---
  }
  if(!keepCore){
    feature_1mer <- feature_1mer[, -(((coreStart-1)*4+1):((coreStart+coreLen-1)*4))]
    feature_2mer <- feature_2mer[, -(((coreStart-1)*16+1):((coreStart+coreLen-2)*16))]
    feature_3mer <- feature_3mer[, -(((coreStart-1)*64+1):((coreStart+coreLen-3)*64))]
  }  
}else{
  ### pre-alloc ...
  n <- nchar(seqs[1])
  m <- length(seqs)
  feature_1mer <- matrix(data=0, nrow=m, ncol=4*n)
  feature_2mer <- matrix(data=0, nrow=m, ncol=16*(n-1))
  feature_3mer <- matrix(data=0, nrow=m, ncol=64*(n-2))
  ### ...
  for (i in 1 : m){
    feature_1mer[i,] <- encode_1mer(seqs[i], map_1mer) + encode_1mer(complDNA(seqs[i]), map_1mer)
    feature_2mer[i,] <- encode_2mer(seqs[i], map_2mer) + encode_2mer(complDNA(seqs[i]), map_2mer)
    feature_3mer[i,] <- encode_3mer(seqs[i], map_3mer) + encode_3mer(complDNA(seqs[i]), map_3mer)    
    # #### --- prepare fasta file
    # cat(paste0('>', i), file=tmpFaFile, append=T, fill=T)
    # cat(seqs[i], file=tmpFaFile, append=T, fill=T)
    # #### ---
  }
  if(keepCore){
    feature_1mer <- feature_1mer[, 1:(n/2*4)]
    feature_2mer <- feature_2mer[, 1:(n/2*16)]
    feature_3mer <- feature_3mer[, 1:((n/2-1)*64)]        
  }else{
    feature_1mer <- feature_1mer[, 1:((n/2-coreLen/2)*4)]
    feature_2mer <- feature_2mer[, 1:((n/2-coreLen/2)*16)]
    feature_3mer <- feature_3mer[, 1:((n/2-coreLen/2)*64)] # n/2-coreLen/2 = (n/2-1)-(coreLen/2-1)
  }
}
## ...


# add interaction into the shape features ...


# write combinations of features to files ++
library(bitops)

## mapping of features to binary barcodes ...
features <- list(feature_1mer)
pool <- c('10000000000')
## ...

## what combinations of features are to be made ...
cmb <- read.table(featureList, header=F, colClasses=c('character', 'character'))
combinations <- cmb[, 2]
#combinations <- c('10000000000', '10011110000', '11000000000', '11011110000', '10100000000', '10111110000', '11100000000')
## ...

if(logarithm){
  y <- log2(signals)
}else{
  y <- signals
}

for(i in combinations){
  pool_bin <- strtoi(pool, base=2)
  cb <- strtoi(i, base=2)
  features_selected <- do.call(cbind, features[which(bitAnd(pool_bin, cb) == pool_bin)])
  features_selected <- cbind(y,rep(c(1), nrow(features_selected)), features_selected) # add y column and constant column
  outFile <- paste0(outFolder, '/', identifier, '.', i)
  write.table(features_selected, outFile, sep=" ", quote=F, row.names=F, col.names=F)  
}  

listFile <- paste0(outFolder, '/list.txt')
cat(identifier, file=listFile, append=T, fill=T)
# ++
