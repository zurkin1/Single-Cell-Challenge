dge.driver.bin <- as.matrix(read.csv(file='c:\\work\\dge_binarized_distMap.csv'))
insitu.bin <- t(as.matrix(read.csv(file='c:\\work\\binarized_bdtnp.csv', check.names = FALSE)))
library('mccr')
actual.pos <- rep(NA, ncol(dge.driver.bin))
for (query.cell in 1:ncol(dge.driver.bin)) {
  mcc <- sapply(1:ncol(insitu.bin), function(i) 
    mccr(dge.driver.bin[, query.cell], insitu.bin[, i]))
  actual.pos[query.cell] <- which.max(mcc)
}
# options(max.print=999999)