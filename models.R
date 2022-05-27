library(InformationValue)
library(PCAmixdata)
library(FactoMineR)
library(regclass)
library(e1071)
library(MASS)
library(glmnet)
library(boot)
library(ggplot2)
library(tidyr)

#quantitative/qualitative features
features <- c(1:13)

#PCAmixdata Procedure
trainsplit <- splitmix(train[,features])
X1 <- trainsplit$X.quanti
X2 <- trainsplit$X.quali
train.pcamix <- PCAmix(X.quanti=X1,X.quali=X2,rename.level=T,graph=T)
train.pcarot <- PCArot(train.pcamix,dim=2,graph=T)

testsplit <-splitmix(test[,features])
X1test <- testsplit$X.quanti
X2test <- testsplit$X.quali
test.pcamix <- predict(train.pcamix,X.quanti=X1test,X.quali=X2test)
test.pcarot <- predict(train.pcarot,X.quanti=X1test,X.quali=X2test)

#factor scores
factors <- c('f_1','f_2','f_3','f_4','f_5')
factors_rot <- c('f_1','f_2')

train_scores <- data.frame(train.pcamix$scores)
test_scores <- data.frame(test.pcamix)
train_scores_rot <- data.frame(train.pcarot$scores)
test_scores_rot <- data.frame(test.pcarot)

#train_scores <- data.frame(heart.famd$ind['coord'])
#test_scores <- data.frame(heart.famd.test$coord)

colnames(train_scores) <- factors
colnames(test_scores) <- factors
colnames(train_scores_rot) <- factors_rot
colnames(test_scores_rot) <- factors_rot

train_scores$target <- train$target
test_scores$target <- test$target
train_scores_rot$target <- train$target
test_scores_rot$target <- test$target

#--------------------------------------------------------
#FAMD + logistic regression models (No Rotation)
#--------------------------------------------------------

#Full Model
full.model <- glm(target ~ .,data=train_scores,family=binomial)
#Reduced Model
reduced.model <- glm(target ~ f_1 + f_2 + f_3 + f_4,data=train_scores,family=binomial)

reduced.predict <- predict(reduced.model,data=train_scores,type="response")

#Find optimal cutoff for cross-validation (Based on Youden's Index)
cutoff <- optimalCutoff(actuals = train_scores$target, predictedScores = reduced.predict,optimiseFor = "Both")
cost <- function(r, pi = 0) mean(abs(r-pi) > cutoff)

#cross-validation (LOOCV)
full.err_ = NULL
reduced.err_ = NULL

full.err_ <- cv.glm(train_scores,full.model,cost=cost)$delta[1]
reduced.err_ <- cv.glm(train_scores,reduced.model,cost=cost)$delta[1]

c(mean(full.err_),mean(reduced.err_))

#--------------------------------------------------------
#FAMD + logistic regression models (Rotation)
#--------------------------------------------------------

#Full Model
full.model.rot <- glm(target ~ .,data=train_scores_rot,family=binomial)
#Reduced Model
reduced.model.rot <- glm(target ~ f_1,data=train_scores_rot,family=binomial)

full.predict.rot <- predict(full.model.rot,data=train_scores_rot,type="response")
reduced.predict.rot <- predict(reduced.model.rot,data=train_scores_rot,type="response")

#Find optimal cutoff for cross-validation (Based on Youden's Index)
cutoff_ <- optimalCutoff(actuals = train_scores_rot$target, predictedScores = reduced.predict.rot,optimiseFor = "Both")
cost <- function(r, pi = 0) mean(abs(r-pi) > cutoff_)

#cross-validation (LOOCV)
full.err = NULL
reduced.err = NULL

#LOOCV

full.err <- cv.glm(train_scores_rot,full.model.rot,cost=cost)$delta[1]
reduced.err <- cv.glm(train_scores_rot,reduced.model.rot,cost=cost)$delta[1]


c(mean(full.err),mean(reduced.err))

#--------------------------------------------------------
# Non Parametric Bootstrap
#--------------------------------------------------------

logit.bootstrap <- function(data,indicies){
  d <- data[indicies,]
  fit <- glm(target ~ .,data=d,family=binomial)
  return(coef(fit))
}
logit.boot <- boot(data=train_scores,statistic=logit.bootstrap,R=1000)


#--------------------------------------------------------
# Accuracy for Unrotated Models
#--------------------------------------------------------

full.predict.test <- predict(full.model,newdata=test_scores,type="response")
confusionMatrix(test_scores$target,full.predict.test,cutoff)

reduced.predict.test <- predict(reduced.model,newdata=test_scores,type="response")
confusionMatrix(test_scores$target,reduced.predict.test,cutoff)

#--------------------------------------------------------
# Accuracy for Rotated Models
#--------------------------------------------------------

full.rot.predict.test <- predict(full.model.rot,newdata=test_scores_rot,type="response")
confusionMatrix(test_scores_rot$target,full.rot.predict.test,cutoff_)

reduced.rot.predict.test <- predict(reduced.model.rot,newdata=test_scores_rot,type="response")
confusionMatrix(test_scores_rot$target,reduced.rot.predict.test,cutoff_)

#-------------------------------------------------------
# Checking For Linearity between predictors and odds
#-------------------------------------------------------

#full model

probabilities <- predict(full.model, type = "response")

# Select only numeric predictors
mydata <- train_scores[,1:5] %>%
  dplyr::select_if(is.numeric) 
predictors <- colnames(mydata)
# Bind the logit and tidying the data for plot
mydata <- mydata %>%
  mutate(logit = log(probabilities/(1-probabilities))) %>%
  gather(key = "predictors", value = "predictor.value", -logit)

ggplot(mydata, aes(logit, predictor.value))+
  geom_point(size = 0.5, alpha = 0.5) +
  geom_smooth(method = "loess") + 
  theme_bw() + 
  facet_wrap(~predictors, scales = "free_y")

#reduced model

probabilities <- predict(reduced.model, type = "response")

# Select only numeric predictors
mydata <- train_scores[,1:4] %>%
  dplyr::select_if(is.numeric) 
predictors <- colnames(mydata)
# Bind the logit and tidying the data for plot
mydata <- mydata %>%
  mutate(logit = log(probabilities/(1-probabilities))) %>%
  gather(key = "predictors", value = "predictor.value", -logit)

ggplot(mydata, aes(logit, predictor.value))+
  geom_point(size = 0.5, alpha = 0.5) +
  geom_smooth(method = "loess") + 
  theme_bw() + 
  facet_wrap(~predictors, scales = "free_y")


#full/reduced model rotated

probabilities <- predict(reduced.model.rot, type = "response")

# Select only numeric predictors
mydata <- train_scores_rot[,1:2] %>%
  dplyr::select_if(is.numeric) 
predictors <- colnames(mydata)
# Bind the logit and tidying the data for plot
mydata <- mydata %>%
  mutate(logit = log(probabilities/(1-probabilities))) %>%
  gather(key = "predictors", value = "predictor.value", -logit)

ggplot(mydata, aes(logit, predictor.value))+
  geom_point(size = 0.5, alpha = 0.5) +
  geom_smooth(method = "loess") + 
  theme_bw() + 
  facet_wrap(~predictors, scales = "free_y")


probabilities <- predict(full.model.rot, type = "response")

# Select only numeric predictors
mydata <- train_scores_rot[,1:2] %>%
  dplyr::select_if(is.numeric) 
predictors <- colnames(mydata)
# Bind the logit and tidying the data for plot
mydata <- mydata %>%
  mutate(logit = log(probabilities/(1-probabilities))) %>%
  gather(key = "predictors", value = "predictor.value", -logit)

ggplot(mydata, aes(logit, predictor.value))+
  geom_point(size = 0.5, alpha = 0.5) +
  geom_smooth(method = "loess") + 
  theme_bw() + 
  facet_wrap(~predictors, scales = "free_y")

#playing around
View(mydata)

x=data.frame(mydata$predictors,mydata$logit,mydata$predictor.value)
hmf=0
hmf2=0
namesty=x$mydata.predictors
for(i in 1:length(mydata$predictors)){
  
  if(namesty[i]=="f_1"){
    hmf=hmf+1
  }
  if(namesty[i]=="f_2"){
    hmf2=hmf2+1
  }
}

f1_predictors=rep(0,hmf)
f1_logit=rep(0,hmf)
f2_predictors=rep(0,hmf2)
f2_logit=rep(0,hmf2)

for(i in 1:length(mydata$predictors)){
  
  if(namesty[i]=="f_1"){
    f1_predictors[i]=x$mydata.predictor.value[i]
    f1_logit[i]=x$mydata.logit[i]
  }
  if(namesty[i]=="f_2"){
    f2_predictors[i-hmf]=x$mydata.predictor.value[i]
    f2_logit[i-hmf]=x$mydata.logit[i]
  }
}

r=randomForest(ifelse(x$mydata.predictors=="f_1",1,0)~., data=x, importance=TRUE,
             proximity=TRUE)

r$importance
r$predicted
r$call
r$mse
r$rsq
r$oob.times
r$importance

library(pROC)
plot.roc(x$mydata.predictors,r$predicted,print.auc=TRUE)

library(ggplot2)
library(ggpubr)
stable=data.frame(treat=f1_logit,control=f1_predictors)

ggballoonplot(stable,fill="value")+scale_fill_viridis_c(option = "C")

c1=chisq.test(f1_logit, f2_logit, correct=TRUE)
c2=chisq.test(f1_predictors, f2_predictors, correct=TRUE)


c3=chisq.test(f1_logit, f2_logit, correct=FALSE)
c4=chisq.test(f1_predictors, f2_predictors, correct=FALSE)


library(BSDA)

z.test(f1_logit,sigma.x=1)
z.test(f1_predictors,sigma.x=1)
z.test(f2_logit,sigma.x=1)
z.test(f2_predictors,sigma.x=1)

nsize(f1_logit,b=1)
nsize(f2_logit,b=1)
nsize(f1_predictors,b=1)
nsize(f2_predictors,b=1)

ntester(f1_logit)
ntester(f2_logit)
ntester(f1_predictors)
ntester(f2_predictors)


library(FSA)
a=z.test(f1_logit,sigma.x=1)
a$p.value
a$statistic


library(sgof)

a=Bayesian.SGoF(f1_logit, alpha = 0.1, gamma = 0.05, P0 = 0.5, a0 = 1, b0 = 1)
a$Rejections
a=Bayesian.SGoF(f2_logit, alpha = 0.1, gamma = 0.05, P0 = 0.5, a0 = 1, b0 = 1)
a$Rejections
a=Bayesian.SGoF(f1_predictors, alpha = 0.1, gamma = 0.05, P0 = 0.5, a0 = 1, b0 = 1)
a$Rejections
a=Bayesian.SGoF(f2_predictors, alpha = 0.1, gamma = 0.05, P0 = 0.5, a0 = 1, b0 = 1)
a$Rejections

a=BY(f1_logit, alpha = 0.05)
a$Adjusted.pvalues
#a=BY(f2_logit, alpha = 0.05)
#a$Adjusted.pvalues
#a=BY(f1_predictors, alpha = 0.05)
#a$Adjusted.pvalues
#a=BY(f2_predictors, alpha = 0.05)
#a$Adjusted.pvalues

Data_Bonferroni =
  p.adjust(a$Adjusted.pvalues,
           method = "bonferroni")

Data_BH =
  p.adjust(a$Adjusted.pvalues,
           method = "BH")

Data_Holm =
  p.adjust(a$Adjusted.pvalues,
           method = "holm")

Data_Hochberg =
  p.adjust(a$Adjusted.pvalues,
           method = "hochberg")

Data_Hommel =
  p.adjust(a$Adjusted.pvalues,
           method = "hommel")

Data_BY =
  p.adjust(a$Adjusted.pvalues,
           method = "BY")

X = a$Adjusted.pvalues
Y = cbind(Data_Bonferroni,
          Data_BH,
          Data_Holm,
          Data_Hochberg,
          Data_Hommel,
          Data_BY)

df=data.frame(Data_Bonferroni,
         Data_BH,
         Data_Holm,
         Data_Hochberg,
         Data_Hommel,
         Data_BY)

matplot(X, Y,
        xlab="Raw p-value",
        ylab="Adjusted p-value",
        type="l",
        asp=1,
        col=1:6,
        lty=1,
        lwd=2)

legend('bottomright',
       legend = c("Bonferroni", "BH", "Holm", "Hochberg", "Hommel", "BY"),
       col = 1:6,
       cex = 1,   
       pch = 16)

abline(0, 1,
       col=1,
       lty=2,
       lwd=1)

u=lm(mydata.logit~mydata.predictor.value,data=x)
summary(u)
plot(u)
View(u)

#library(keras)

hist(f1_predictors)
hist(f2_predictors)
hist(f1_logit)
hist(f2_logit)
#keras_model(x$mydata.predictor.value,x$mydata.logit)

cor(ifelse(x$mydata.predictors=="f_1",0,1),x$mydata.logit)
cor(ifelse(x$mydata.predictors=="f_1",0,1),x$mydata.predictor.value)
u=lm(ifelse(x$mydata.predictors=="f_1",0,1)~mydata.predictor.value,data=x)
plot(u)
summary(u)
u=lm(ifelse(x$mydata.predictors=="f_1",0,1)~mydata.logit,data=x)
plot(u)
summary(u)

ggplot(x,aes(ifelse(mydata.predictors=="f_1",0,1),mydata.logit))+geom_point()
ggplot(x,aes(ifelse(mydata.predictors=="f_1",0,1),mydata.predictor.value))+geom_point()

