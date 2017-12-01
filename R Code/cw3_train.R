#Set the directory
setwd("./hw/ml/CW3")  
Sys.setenv("JAVA_HOME"="C:\\Program Files\\Java\\jre1.8.0_151")

library(ggplot2)
library(RWeka)
library(caret)




#Read Dataset
happy_test<- read.csv("./Dataset/fer2017-testing-happy.csv")
happy_train<- read.csv("./Dataset/fer2017-training-happy.csv")
test<- read.csv("./Dataset/fer2017-testing.csv")
train<- read.csv("./Dataset/fer2017-training.csv")

#Pre Processing
table(as.factor(train$emotion))

ggplot(train,aes(x=as.factor(emotion),fill=emotion))+
  geom_bar(stat="count",color="black")+
  scale_fill_gradient(low="lightgreen",high="lightblue",guide=FALSE)+
  labs(title="Emotion in Train Data",x="emotions")



table(as.factor(train$emotion))



sample <- sample(1:nrow(train),50)
var <- t(train[sample,-1])
var_matrix <- lapply(1:50,function(x) matrix(var[,x],ncol=48))
opar <- par(no.readonly = T)
par(mfrow=c(5,10),mar=c(.1,.1,.1,.1))

for(i in 1:50) {
  for(j in 1:48) {
    var_matrix[[i]][j,] <- rev(var_matrix[[i]][j,])
  }
  image(var_matrix[[i]],col=grey.colors(225),axes=F)
}


#Unique records
train<-unique(train)
table(as.factor(train$emotion))

ggplot(train,aes(x=as.factor(emotion),fill=emotion))+
  geom_bar(stat="count",color="black")+
  scale_fill_gradient(low="lightgreen",high="lightblue",guide=FALSE)+
  labs(title="Emotion in Train Data",x="emotions")



table(as.factor(train$emotion))



sample <- sample(1:nrow(train),50)
var <- t(train[sample,-1])
var_matrix <- lapply(1:50,function(x) matrix(var[,x],ncol=48))
opar <- par(no.readonly = T)
par(mfrow=c(5,10),mar=c(.1,.1,.1,.1))

for(i in 1:50) {
  for(j in 1:48) {
    var_matrix[[i]][j,] <- rev(var_matrix[[i]][j,])
  }
  image(var_matrix[[i]],col=grey.colors(225),axes=F)
}

#Look for Near Zero Variance
nzr <- nearZeroVar(train[,-1],saveMetrics=T,freqCut=10000/1,uniqueCut=1/7)
sum(nzr$zeroVar)
sum(nzr$nzv)

#PCA

emotion <- as.factor(train[[1]])
train$emotion <- NULL
train <- train/255
covtrain <- cov(train)


train_pc <- prcomp(covtrain)
varex <- train_pc$sdev^2/sum(train_pc$sdev^2)
varcum <- cumsum(varex)
result <- data.frame(num=1:length(train_pc$sdev),
                     ex=varex,
                     cum=varcum)
plot(result$num,result$cum,type="b",xlim=c(0,100),
     main="Variance Explained by Top 100 Components",
     xlab="Number of Components",ylab="Variance Explained")
abline(v=25,lty=2)


train_score <- as.matrix(train) %*% train_pc$rotation[,1:25]
train <- cbind(emotion,as.data.frame(train_score))

dim(train)
table(as.factor(train$emotion))

colors <- rainbow(length(unique(train$emotion)))
names(colors) <- unique(train$emotion)
plot(train$PC1,train$PC2,type="n",main="First Two Principal Components")
text(train$PC1,train$PC2,label=train$emotion,col=colors[train$emotion])

write.csv(train,"./ReducedDataset/train.csv")
write.arff(train,"./ReducedDataset/train.arff")

#J48 classifier
resultJ48 <- J48(emotion~., train)
summary(resultJ48)


library(rpart)
resultDT<- rpart(emotion~.,train)
