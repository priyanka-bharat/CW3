#Set the directory
setwd("./hw/ml/CW3")

#Read Dataset
test<- read.csv("./Dataset/fer2017-testing.csv")

#Pre Processing
table(as.factor(test$emotion))

ggplot(test,aes(x=as.factor(emotion),fill=emotion))+
  geom_bar(stat="count",color="black")+
  scale_fill_gradient(low="lightgreen",high="lightblue",guide=FALSE)+
  labs(title="Emotion in test Data",x="emotions")



table(as.factor(test$emotion))



sample <- sample(1:nrow(test),50)
var <- t(test[sample,-1])
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
test<-unique(test)
table(as.factor(test$emotion))

ggplot(test,aes(x=as.factor(emotion),fill=emotion))+
  geom_bar(stat="count",color="black")+
  scale_fill_gradient(low="lightgreen",high="lightblue",guide=FALSE)+
  labs(title="Emotion in test Data",x="emotions")



table(as.factor(test$emotion))



sample <- sample(1:nrow(test),50)
var <- t(test[sample,-1])
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
nzr <- nearZeroVar(test[,-1],saveMetrics=T,freqCut=10000/1,uniqueCut=1/7)
sum(nzr$zeroVar)
sum(nzr$nzv)

#PCA

emotion <- as.factor(test[[1]])
test$emotion <- NULL
test <- test/255
covtest <- cov(test)


test_pc <- prcomp(covtest)
varex <- test_pc$sdev^2/sum(test_pc$sdev^2)
varcum <- cumsum(varex)
result <- data.frame(num=1:length(test_pc$sdev),
                     ex=varex,
                     cum=varcum)
plot(result$num,result$cum,type="b",xlim=c(0,100),
     main="Variance Explained by Top 100 Components",
     xlab="Number of Components",ylab="Variance Explained")
abline(v=25,lty=2)


test_score <- as.matrix(test) %*% test_pc$rotation[,1:25]
test <- cbind(emotion,as.data.frame(test_score))

dim(test)
table(as.factor(test$emotion))

colors <- rainbow(length(unique(test$emotion)))
names(colors) <- unique(test$emotion)
plot(test$PC1,test$PC2,type="n",main="First Two Principal Components")
text(test$PC1,test$PC2,label=test$emotion,col=colors[test$emotion])


write.csv(test,"./ReducedDataset/test.csv")
write.arff(test,"./ReducedDataset/test.arff")


#J48 classifier
resultJ48 <- J48(emotion~., test)
summary(resultJ48)
plot(resultJ48)
