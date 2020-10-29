library(keras)
library(caret)
library(mccr)
# Predict geometry coordinate by neural network

#Normalize data
data <- t(dm@data)

non.zero.prop <- apply(data, 2, function(y) sum(y != 0)/length(y))

data <- data[,non.zero.prop>0.]

data <- apply(data, 2, function(x) (x - min(x))/(max(x) - min(x))  )

pos <- sapply(1:nrow(data), function(i) which.max(dm@mcc.scores[,i]))

y <- dm@geometry[pos,]

x <- data

#Build model
model <- keras_model_sequential() %>%
  layer_dense(units  = 1000
              ,input_shape = ncol(x)
              ,kernel_regularizer = regularizer_l2(l = 0.001)) %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units  = 500
              ,activation  = 'relu'
              ,kernel_regularizer = regularizer_l2(l = 0.001)) %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units  = 100
              ,activation  = 'relu'
              ,kernel_regularizer = regularizer_l2(l = 0.001)) %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units  = ncol(y))

model %>% compile(
  loss      = 'mse',
  optimizer ='adam',
)

#K-fold CV with k = 5
folds <- createFolds(1:nrow(x), k = 5)
for (i in 1:length(folds))
{
  idx <- folds[[i]]
  train <- x[-idx,]
  train.y <- y[-idx,]
  
  test <- x[idx,]
  test.y <- y[idx,]
  
  history = model %>% fit(
    train, train.y, 
    epochs = 10,
    steps_per_epoch = 50,
    validation_data = list(test, test.y),
    validation_steps = 1
  )
}

#Predict geometry coordinate on all data
y_pred <- model %>% predict(x)

#Clustering on predicted coordinate with k = 9
kp <- kmeans(y_pred, 9, nstart = 1000, iter.max = 5000)

#Gene ranking
bin.gene <- rownames(dm@binarized.data)

data.84gene <- t(dm@data[bin.gene,])

#Anova for each genes
p <- c()
for (i in 1:84) {
  fit <- fit <- lm(formula = data.84gene[,i] ~ as.factor(kp$cluster))
  
  p <- c(p, anova(fit)$`Pr(>F)`[1])
}

#Ranking based on p-value
r <- rank(p)

#Ranking based on variance of binarized data
r1 <- 85 - rank(colSds(t(dm@binarized.data)))

#Average rank
final.r <- (r+r1)/2

#Selected gene
select.gene <- bin.gene[order(final.r)[1:60]]

#get top 10 position based on geometry prediction

top10geo <- function(y_pred, geo)
{
  tmp1 <- geo[1:3039,]
  result <- matrix(ncol = 10, nrow = nrow(y_pred))
  
  for (i in 1:nrow(y_pred)) {
    mse <- colSums((y_pred[i,] - t(tmp1))^2)
    or <- order(mse)
    result[i,] <- or[1:10]
  }
  result
}

#get top 10 position based on maximal mcc
bi.data <- t(dm@binarized.data)
insitu.data <- dm@insitu.matrix

top10mcc <- function(genes, bi.data, insitu.data)
{
  tmp1 <- bi.data[,genes]
  tmp2 <- insitu.data[,genes]
  result <- matrix(nrow = nrow(bi.data), ncol = 10)
  
  for (i in 1:nrow(bi.data)) {
    tmp3 <- c()
    for (j in 1:nrow(insitu.data)) {
      tmp3 <- c(tmp3, mccr(tmp1[i,], tmp2[j,]))
    }
    result[i,] <- order(tmp3, decreasing = TRUE)[1:10]
  }
  result
}


for (i in c(60,40,20)) {
  geo <- top10geo(y_pred, dm@geometry)
  mcc <- top10mcc(select.gene[1:i], bi.data, insitu.data)
  
  select.pos <- matrix(nrow = nrow(geo), ncol = 10)
  for (j in 1:nrow(geo)) {
    to.add <- intersect(geo[j,], mcc[j,])
    n <- round((10 - length(to.add))/2)
    to.add <- c(to.add, geo[j,][which(!geo[j,] %in% to.add )][1:n])
    n <- 10 - length(to.add)
    to.add <- c(to.add, mcc[j,][which(!mcc[j,] %in% to.add )][1:n])
    select.pos[j,] <- to.add
  }
  write.csv(matrix(select.gene[1:i], ncol = 10), file = paste0("Submit/gene",i,".csv"))
  write.csv(select.pos, file = paste0("Submit/pos",i,".csv"))
}







