

#Set the directory
setwd("./hw/ml/CW3")  
Sys.setenv("JAVA_HOME"="C:\\Program Files\\Java\\jre1.8.0_151")

library(ggplot2)
library(RWeka)
library(caret)


#Read Dataset
happy_test<- read.csv("./Dataset/fer2017-testing-happy.csv")


#Pre Processing
table(as.factor(happy_test$emotion))

ggplot(happy_test,aes(x=as.factor(emotion),fill=emotion))+
  geom_bar(stat="count",color="black")+
  labs(title="Emotion in happy_test Data",x="emotions")



table(as.factor(happy_test$emotion))



sample <- sample(1:nrow(happy_test),50)
var <- t(happy_test[sample,-1])
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
happy_test<-unique(happy_test)
table(as.factor(happy_test$emotion))

ggplot(happy_test,aes(x=as.factor(emotion),fill=emotion))+
  geom_bar(stat="count",color="black")+
  labs(title="Emotion in happy_test Data",x="emotions")



table(as.factor(happy_test$emotion))



sample <- sample(1:nrow(happy_test),50)
var <- t(happy_test[sample,-1])
var_matrix <- lapply(1:50,function(x) matrix(var[,x],ncol=48))
opar <- par(no.readonly = T)
par(mfrow=c(5,10),mar=c(.1,.1,.1,.1))

for(i in 1:5) {
  for(j in 1:48) {
    var_matrix[[i]][j,] <- rev(var_matrix[[i]][j,])
  }
  image(var_matrix[[i]],col=grey.colors(225),axes=F)
}

#Look for Near Zero Variance
nzr <- nearZeroVar(happy_test[,-1],saveMetrics=T,freqCut=10000/1,uniqueCut=1/7)
sum(nzr$zeroVar)
sum(nzr$nzv)

#PCA

emotion <- as.factor(happy_test[[1]])
happy_test$emotion <- NULL
happy_test <- happy_test/255
covhappy_test <- cov(happy_test)


happy_test_pc <- prcomp(covhappy_test)
varex <- happy_test_pc$sdev^2/sum(happy_test_pc$sdev^2)
varcum <- cumsum(varex)
result <- data.frame(num=1:length(happy_test_pc$sdev),
                     ex=varex,
                     cum=varcum)
plot(result$num,result$cum,type="b",xlim=c(0,100),
     main="Variance Explained by Top 100 Components",
     xlab="Number of Components",ylab="Variance Explained")
abline(v=25,lty=2)


happy_test_score <- as.matrix(happy_test) %*% happy_test_pc$rotation[,1:25]
happy_test <- cbind(emotion,as.data.frame(happy_test_score))

dim(happy_test)
table(as.factor(happy_test$emotion))

colors <- rainbow(length(unique(happy_test$emotion)))
names(colors) <- unique(happy_test$emotion)
plot(happy_test$PC1,happy_test$PC2,type="n",main="First Two Principal Components")
text(happy_test$PC1,happy_test$PC2,label=happy_test$emotion,col=colors[happy_test$emotion])

write.csv(happy_test,"./ReducedDataset/happy_test.csv")
write.arff(happy_test,"./ReducedDataset/happy_test.arff")

#J48 classifier
resultJ48 <- J48(emotion~., happy_test)
summary(resultJ48)


library(rpart)
resultDT<- rpart(emotion~.,happy_test)
