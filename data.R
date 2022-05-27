library(dplyr)

set.seed(13)

#load data
#heart <- read.csv('Z:/Data/heart_disease/heart.csv',header=TRUE)
#heart <- subset(heart,select = -c(restecg,fbs,trestbps))

heart$sex <- recode_factor(heart$sex,"1"="Male","0"="Female")
heart$cp <- recode_factor(heart$cp,"0"="Typical Angina","1"="Atypical Angina","2"="Non-Anglinal Pain","3"="Asymptomatic")
heart$restecg <- recode_factor(heart$restecg,"0"="Normal","1"="ST-T Wave Abnormality","2"="Positive Left Ventricular Hypertrophy")
heart$exang <- recode_factor(heart$exang,"1"="EIA","0"="No EIA")
heart$ca <- recode_factor(heart$ca,"0"="0 MVC","1"="1 MVC","2"="2 MVC","3"="3 MVC","4"="4 MVC")
heart$thal <- recode_factor(heart$thal,"1"="Normal","2"="Fixed Defect","3"="Reversable Defect")
heart$slope <- recode_factor(heart$slope,"0"="Upsloping","1"="Flat","2"="Downsloping")
heart$fbs <- recode_factor(heart$fbs,"0"="FBS < 120 mg/dl","1"="FBS > 120 mg/dl")

#train/test split (80%/20%)
train_index <- sample(1:nrow(heart),0.80*nrow(heart))
test_index <- setdiff(1:nrow(heart),train_index)
train <- heart[train_index,]
test <- heart[test_index,]

#quantitative and qualitative variables 
quant <- c('trestbps','age','chol','thalach','oldpeak')
#qual <- c('sex','ca','exang','thal','slope')
qual <- c('sex','restecg','ca','exang','thal','slope','fbs')

#tidying up names and data types
names(train)[1]<-'age'
for (name in qual)
{
  data <- unlist(train[name],use.names=FALSE)
  train[name] <- factor(data)
}

names(test)[1]<-'age'
for (name in qual)
{
  data <- unlist(test[name],use.names=FALSE)
  test[name] <- factor(data)
}

