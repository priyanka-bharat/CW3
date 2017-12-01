#Set the directory
setwd("./hw/ml/CW3")

train<- read.csv("./Dataset/fer2017-training.csv")

displayEmotion <- function(X){
  m <- matrix(unlist(X),nrow = 48,byrow = T)
  m <- t(apply(m, 2, rev))
  image(m,col=grey.colors(255))
}

displayEmotion(train[9,-1])

nzv <- nearZeroVar(train)
nzv.nolabel <- nzv-1

inTrain <- createDataPartition(y=train$emotion, p=0.7, list=F)

training <- train[inTrain, ]
CV <- train[-inTrain, ]

X <- as.matrix(training[, -1]) # data matrix (each row = single example)
N <- nrow(X) # number of examples
y <- training[, 1] # class labels
K <- length(unique(y)) # number of classes
X.proc <- X/max(X) # scale
D <- ncol(X.proc) # dimensionality
Xcv <- as.matrix(CV[, -1]) # data matrix (each row = single example)
ycv <- CV[, 1] # class labels
Xcv.proc <- Xcv/max(X) # scale CV data
Y <- matrix(0, N, K)

for (i in 1:N){
  Y[i, y[i]+1] <- 1
}

nnet.mnist <- nnet(X.proc, Y, step_size = 0.3, reg = 0.0001, niteration = 3500)
predicted_class <- nnetPred(X.proc, nnet.mnist)
print(paste('training set accuracy:',mean(predicted_class == (y+1))))
predicted_class <- nnetPred(Xcv.proc, nnet.mnist)
print(paste('CV accuracy:',mean(predicted_class == (ycv+1))))
Xtest <- Xcv[sample(1:nrow(Xcv), 1), ]
Xtest.proc <- as.matrix(Xtest[-nzv.nolabel], nrow = 1)
predicted_test <- nnetPred(t(Xtest.proc), nnet.mnist)
print(paste('The predicted digit is:',predicted_test-1 ))
displayDigit(Xtest)
