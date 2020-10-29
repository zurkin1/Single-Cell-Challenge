rm(list=ls())
library(randomForest)
library(mccr)
library(foreach)
library(doParallel)
library(doRNG)
library(caret)

ncores=10  #number of cores used for 20-fold cross validation use a number from 1 to 20


#load data
#read raw data
geometry = read.table("geometry.txt",sep = " ",header=TRUE)
colnames(geometry) = c("x","y","z")

insitu.matrix = read.table("binarized_bdtnp.csv.gz", sep = ",",header = T)
insitu.genes_orig <- colnames(insitu.matrix)
insitu.genes = gsub(".","-",insitu.genes_orig,fixed = T)
# also replace .spl. --> (spl)
insitu.genes = gsub("-spl-","(spl)",insitu.genes,fixed = T)
insitu.bin = as.matrix(insitu.matrix)
colnames(insitu.bin) = insitu.genes
row.names(insitu.bin)<-1:dim(insitu.bin)[1]


dge.bin=t(read.csv("dge_binarized_distMap.csv"))

dge.norm=t(read.table("dge_normalized.txt"))

### end data reading






#define geometric location of bins with highest mcc score between SC expression and insitu data for all 84 genes
actual.xyz<-matrix(NA, nrow(dge.bin),3)
actual.pos <- rep(NA, nrow(dge.bin))
for (query.cell in 1:nrow(dge.bin)) {
  mcc <- sapply(1:nrow(insitu.bin), function(i) 
    mccr(dge.bin[query.cell,], insitu.bin[i,]))
  geom=geometry[mcc==max(mcc),]
  actual.xyz[query.cell,]<-apply(geom,2,mean)
}
colnames(actual.xyz)<-c("x","y","z")
Y=actual.xyz




#Step 1
#use 20-fold cross validation to fit xyx coordinates on all 84 genes from dge.bin and get the predicted xyz locations

set.seed(1)
folds<-caret::createFolds(1:dim(dge.bin)[1],k=20)
cl <- makePSOCKcluster(ncores)
registerDoParallel(cl)


Ypcvv=foreach(j=1:length(folds),.combine="rbind") %dorng%{
  Yp=matrix(NA,length(folds[[j]]),3)
  cat(j);cat("\n")
  for(i in 1:3){ #for each coordinate
    mod=randomForest::randomForest(dge.bin[-folds[[j]],insitu.genes],Y[-folds[[j]],i],ntree=2000,importance=FALSE,node.size=2,corr.bias=TRUE)
    Yp[,i]=predict(mod,dge.bin[folds[[j]],insitu.genes],type="response")
  }
  Yp
}

Yp=Ypcvv[order(unlist(folds)),]  
plot(Y,Yp)
sqrt(mean(abs(Y-Yp)^2))

stopImplicitCluster()

########predicted position of 1297 cells is done
##Yp is are the predicted xyz coordinates of the 1297 cells obtained by cross-validation 



#Step 2
#fit best rf model for each coordinate to determine gene importance on dge.bin
set.seed(1)
modlist=list()
#use dge.bin to fit actual coordinates 
for(i in 1:3){
  mod=randomForest(dge.norm[,insitu.genes],Y[,i],ntree=2000,importance=TRUE)
  modlist[[i]]<-mod
}



#determine importance of each coordinate in distance calculations
sds=apply(Y,2,sd)
imp=sds/sum(sds)

#rank insitu genes based on their importance in random forests for each coordinate
ranks=cbind(rank(-importance(modlist[[1]])[insitu.genes,"IncNodePurity"]),
            rank(-importance(modlist[[2]])[insitu.genes,"IncNodePurity"]),
            rank(-importance(modlist[[3]])[insitu.genes,"IncNodePurity"]))
weigthedsumrank=apply(ranks,1,function(x){sum(x*imp)})
orderedinsitus=insitu.genes[order(weigthedsumrank)]





#Step 3
#rank bins by mcc and by distance from the predicted location, and pick top 10

#calculate distance from predicted values. This stays the same for all sub-challenges since no insitu data is used
output.xyz=matrix(NA,dim(dge.bin)[1],dim(insitu.bin)[1])
for(i in 1:dim(dge.bin)[1]){
  dis=apply((geometry-matrix(rep(Yp[i,],each=dim(geometry)[1]),dim(geometry)[1],3))^2,1,mean)
  ch=order(dis)
  output.xyz[i,]<-ch
}





siz=c(60,40,20)

res=NULL
for(kk in 1:length(siz)){ # for each subchallenge
  nG=siz[kk]
  sel=orderedinsitus[1:nG]
  alpha=c(0.05,0.15,0.5)[kk] #weight given to bins based on their proximilty to the predicted location; (1-alpha) is for the mcc score
  
  ##  calculate MCC with the top nG genes 
  out <- matrix(NA, nrow(dge.bin),nrow(insitu.bin))
  for (query.cell in 1:nrow(dge.bin[,sel])) {
    mat.nn= insitu.bin[output.xyz[query.cell,],sel]
    mcc <- sapply(1:nrow(mat.nn), function(i) 
      mccr(dge.bin[query.cell,sel], mat.nn[i,]))
    out[query.cell,] <- rownames(mat.nn)[order(mcc,decreasing=TRUE)]
  }
  out.bin= apply(out,2,function(x){as.numeric(as.character(x))})
  
  
  ### 10 predictions # combine rank of bins based on largest mcc values and closest distance from predicted location
  predictions= matrix(NA, nrow(output.xyz),10)
  for ( pi in 1:nrow(predictions))
  {
    locids<- output.xyz[pi,]
    ranks.xyz<- match(locids,output.xyz[pi,])
    ranks.exp<- match(locids,out.bin[pi,])
    weightedranks<-(alpha*ranks.xyz) + ((1-alpha)*ranks.exp)
    names(weightedranks)<- locids
    orderedranks<-weightedranks[order(weightedranks,decreasing=F)[1:10]]
    predictions[pi,]<- names(orderedranks)
  }
  
  predictions= apply(predictions,2,function(x){as.numeric(as.character(x))})
 
  
  
  
  ###read submisison template and write prediction files
  
  a=read.table("DREAM_SCTS_TeamX_60genes.csv",sep=",",header=FALSE)
  a[is.na(a$V1),"V1"]<-""
  a=as.matrix(a)
  for(ser in 1:6){
    if(ser<=nG/10){
      a[ser,2:11]<-sel[((ser-1)*10+1):((ser-1)*10+10)]
    }else{
      a[ser,2:11]<-"NA"
    }
  }
  a[(which(a[,1]=="1")):(which(a[,1]=="1297")),2:11]<-predictions
  
  write.table(a,sep=",",row.names = F,col.names = F,file=paste("DREAM_SCTS_bcbu_",nG,"genes.csv",sep=""))
  
}






