rsqrd <- function(observe, model){
  ss_res <- sum((observe-model)^2)
  y_bar <- mean(observe)
  ss_tot <- sum((observe-y_bar)^2)
  return (1-ss_res/ss_tot)
}
################################ 

args <- commandArgs(trailingOnly = TRUE)

#prefix <- '../tianyin-method/data/GATA/GATA3_TCATGC20NCG_GATA_10'
#outFile <- '../output/GATA/summary.txt'

prefix <- args[1]
outPath <- args[2]
featureList <- args[3]

outFile <- paste0(outPath, 'summary_r2.txt') # fixed filename "summary.txt"
# settings
## what combinations of features are to be made ...
cmb <- read.table(featureList, header=F, colClasses=c('character', 'character'))
combinations <- cmb[, 2]
#combinations <- c('10000000000', '10011110000', '11000000000', '11011110000', '10100000000', '10111110000', '11100000000')
## ...
subs <- c('.fold01','.fold02','.fold03','.fold04','.fold05','.fold06','.fold07','.fold08','.fold09','.fold10')

r2 <- c()
crr_r2 <- c()

for (i in 1 : nrow(cmb)){
    cb <- cmb[i,2]
    title <- cmb[i,1]
    test <- c()
    predict <- c()
    for (sb in subs){
    	t <- as.matrix(read.table(paste0(prefix, '.', cb, sb, '.test.y'), header=F, colClasses='numeric'))
    	p <- as.matrix(read.table(paste0(prefix, '.', cb, sb, '.test.p'), header=F, colClasses='numeric'))
	test <- c(test, t)
	predict <- c(predict, p)
    }
    crr_r2 <- round(rsqrd(test, predict), 4)
    r2 <- c(r2, crr_r2)
    seqCount <- length(test)
    jpeg(paste0(outPath, basename(prefix), '_', cb, '.jpg'), width=6, height=6, units='in', res=600)
    plot(predict, test, main=paste0(title, " R2: ", round(crr_r2, 4)), sub=paste0("#seq: ", seqCount), xlab="predicted", ylab="experimental")
    dev.off()
    outFile2 <- paste0(outPath, basename(prefix), '_', cb, '_output.txt')
    write(test, outFile2, ncolumns = length(test))
    write(predict, outFile2, append = T,ncolumns = length(predict))
}

num_seq <- length(predict)
r2 <- c(num_seq, r2)
identifier <- basename(prefix)
r2 <- c(identifier, r2)
write(r2, outFile, append=T, ncolumns=length(r2))
