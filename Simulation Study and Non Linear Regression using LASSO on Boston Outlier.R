library(MASS)
library(ggplot2)
library(GGally)
library(tidyr)
library(tidyverse)
library(dplyr)
require(RColorBrewer)
install.packages("randomcoloR")
require(randomcoloR)
data("Boston")
Boston<-as.tibble(Boston)

BostonWithFactors<-Boston
BostonWithFactors$chas<-as.factor(BostonWithFactors$chas)
BostonWithFactors$rad<-as.factor(BostonWithFactors$rad)

smobj<-summary(BostonWithFactors)

path<-"C:/Users/husai/OneDrive - University of Cincinnati/BANA Coursework/Sem2/BANA 7046-002 Data Mining 1/HW2/"
setwd(path)
outliers<-readxl::read_xlsx("BostonOutliers.xlsx")

head(Boston)
write.csv(Boston, "C:/Users/bhavy/OneDrive/Desktop/Information Systems/Semester2/Data Mining/Homework/Homework#2/BostonOutliers.csv")
outliers<-readxl::read_xlsx("BostonOutliers.xlsx")


outliers<-outliers%>%
  mutate(UpperLimit = Median + 1.5* (`3rd Quartile`-`1st Quartile`))%>%
  mutate(LowerLimit = Median - 1.5* (`3rd Quartile`-`1st Quartile`))

UpperOutliers<-c(sum(BostonWithFactors[(outliers$Variable)]$age>outliers$UpperLimit[1]),
                 sum(BostonWithFactors[(outliers$Variable)]$black>outliers$UpperLimit[2]),
                 sum(BostonWithFactors[(outliers$Variable)]$crim>outliers$UpperLimit[3]),
                 sum(BostonWithFactors[(outliers$Variable)]$dis>outliers$UpperLimit[4]),
                 sum(BostonWithFactors[(outliers$Variable)]$indus>outliers$UpperLimit[5]),
                 sum(BostonWithFactors[(outliers$Variable)]$lstat>outliers$UpperLimit[6]),
                 sum(BostonWithFactors[(outliers$Variable)]$medv>outliers$UpperLimit[7]),
                 sum(BostonWithFactors[(outliers$Variable)]$nox>outliers$UpperLimit[8]),
                 sum(BostonWithFactors[(outliers$Variable)]$ptratio>outliers$UpperLimit[9]),
                 sum(BostonWithFactors[(outliers$Variable)]$rm>outliers$UpperLimit[10]),
                 sum(BostonWithFactors[(outliers$Variable)]$tax>outliers$UpperLimit[11]),
                 sum(BostonWithFactors[(outliers$Variable)]$zn>outliers$UpperLimit[12]))

LowerOutliers<-c(sum(BostonWithFactors[(outliers$Variable)]$age<outliers$LowerLimit[1]),
                 sum(BostonWithFactors[(outliers$Variable)]$black<outliers$LowerLimit[2]),
                 sum(BostonWithFactors[(outliers$Variable)]$crim<outliers$LowerLimit[3]),
                 sum(BostonWithFactors[(outliers$Variable)]$dis<outliers$LowerLimit[4]),
                 sum(BostonWithFactors[(outliers$Variable)]$indus<outliers$LowerLimit[5]),
                 sum(BostonWithFactors[(outliers$Variable)]$lstat<outliers$LowerLimit[6]),
                 sum(BostonWithFactors[(outliers$Variable)]$medv<outliers$LowerLimit[7]),
                 sum(BostonWithFactors[(outliers$Variable)]$nox<outliers$LowerLimit[8]),
                 sum(BostonWithFactors[(outliers$Variable)]$ptratio<outliers$LowerLimit[9]),
                 sum(BostonWithFactors[(outliers$Variable)]$rm<outliers$LowerLimit[10]),
                 sum(BostonWithFactors[(outliers$Variable)]$tax<outliers$LowerLimit[11]),
                 sum(BostonWithFactors[(outliers$Variable)]$zn<outliers$LowerLimit[12]))

outliermatrix<-as.data.frame(cbind(UpperOutliers,LowerOutliers,outliers$Variable))

names(outliermatrix)[3]<-"VariableName"

outliermatrix

BostonScaled<-as.tibble(scale(Boston))
BostonScaled$chas<-Boston$chas
BostonScaled$rad<-Boston$rad
dt.long <- gather(BostonScaled, "variable", "value", crim:medv)

ViolinPlotAll<-ggplot(dt.long,aes(factor(variable), value))+
  geom_violin(aes(fill=factor(variable)))+
  geom_boxplot(alpha=0.3, color="black", width=.1)+
  labs(x = "", y = "")+
  theme_minimal()+
  theme(legend.position = "none")+
  facet_wrap(~variable, scales="free",ncol=3)

BoxPlotAll<-ggplot(dt.long,aes(factor(variable), value))+
  geom_boxplot(alpha=0.3, color="black", width=.1)+
  labs(x = "", y = "")+
  theme_minimal()+
  theme(legend.position = "none")+
  facet_wrap(~variable, scales="free",ncol=3)

HistAll<-ggplot(dt.long,aes(x=value))+
  geom_histogram(aes(fill=factor(variable),y=..density..),color="black")+
  geom_density(color="black")+
  labs(x = "", y = "")+
  theme_minimal()+
  theme(legend.position = "none")+
  facet_wrap(~variable, scales="free",ncol=3)

ScatterAll<-BostonScaled %>%
  gather(., "variable", "value", crim:lstat)%>% 
  ggplot(aes(x = value, y = medv)) +
  geom_point(aes(color=as.factor(variable)),alpha=0.25) +
  facet_wrap(~ variable, scales = "free",ncol=3) +
  theme_minimal()+
  geom_smooth(method="lm", linetype=2, se=F, color="black")+
  theme(legend.position = "none")

PPCor<-ggcorr(BostonScaled[1:14],method = c("pairwise", "pearson"),low = "blue", mid = "white", high = "red",label = TRUE,label_round = 2)+
  ggtitle("Pairwise Pearson - Correlation HeatMap") + 
  theme_minimal()

set.seed(12698282)
sample_index<-sample(nrow(BostonScaled),ceiling(nrow(BostonScaled)*0.75))
boston_train<-BostonScaled[sample_index,]
boston_test<-BostonScaled[-sample_index,]

fit<-boston_train%>%
  lm(data=.,medv~.)

summary(fit)

library(leaps)

subset_result <- regsubsets(medv~.,data=boston_train, nbest=1, nvmax = 14)
summary(subset_result)

plot(subset_result, scale=c("adjr2"))

plot(subset_result$cp)
index<-1:14
cbind.data.frame(subset_result$rss,index)
residualplot<-ggplot(data=cbind.data.frame(subset_result$rss,index),aes(x=index,y=subset_result$rss))+
  geom_point()+theme_minimal()+ylab("Residuals")+xlab("")+geom_label(aes(label=index))


set.seed(12698282)
nullmodel<-lm(medv~1, data=boston_train)
fullmodel<-lm(medv~., data=boston_train)
model_step_b <- step(fullmodel,direction='backward')
model_step_f <- step(nullmodel, scope=list(lower=nullmodel, upper=fullmodel), direction='forward')

summary(model_step_b)
summary(model_step_f)


boston_ful<-lm(model_step_f)%>%
  broom::augment() %>%
  mutate(row_num = 1:n())

sm<-summary(model_step_f)

residuals<-as.data.frame(model_step_f$residuals)
names(residuals)<-"Residuals"


norm_res<-(residuals$Residuals - mean(residuals$Residuals))/sd(residuals$Residuals)
yhat<-boston_ful$.fitted

p1<-ggplot(data= residuals,aes(x=Residuals,fill=Residuals)) + theme_minimal() +
  geom_histogram(alpha=0.5,bins=15) 

p2<-ggplot(data= residuals,aes(x=Residuals, fill=Residuals)) + theme_minimal() +
  geom_density(aes(fill=Residuals))

p3<-ggplot(data=as.data.frame(cbind(norm_res,yhat)), aes(x=yhat,y=norm_res))+geom_point(alpha = 0.75,pch=20)+ theme_minimal()+ylab("Standardized Residuals") +
  xlab("Y-hat") +  geom_abline(intercept = 0, slope = 0, color='red',linetype="dotted")+geom_smooth(se=F,linetype="dotted")

p4<-ggQQ(model_step_f)+theme_minimal()+xlab("Theoretical Quantiles")+ylab("Sample")

ggQQ <- function(LM) # argument: a linear model
{
  y <- quantile(LM$resid[!is.na(LM$resid)], c(0.25, 0.75))
  x <- qnorm(c(0.25, 0.75))
  slope <- diff(y)/diff(x)
  int <- y[1L] - slope * x[1L]
  p <- ggplot(LM, aes(sample=.resid)) +
    stat_qq(alpha = 0.5) +
    geom_abline(slope = slope, intercept = int, color="blue",linetype="dotted")
  return(p)
}

gridExtra::grid.arrange(p1, p2, p3,p4, ncol = 2)

library(glmnet)
lasso_fit<- glmnet(x = as.matrix(boston_train[-14]), y = as.matrix(boston_train[14]), alpha = 1)
smlf<-summary(lasso_fit)

cv_lasso_fit <- cv.glmnet(x = as.matrix(boston_train[-14]), y = as.matrix(boston_train[14]), alpha = 1, nfolds = 5)
plot(cv_lasso_fit)

plotlasso<-cbind.data.frame(lasso_fit$a0,lasso_fit$df,log(lasso_fit$lambda))
names(plotlasso)<-c("Mean-Squared Error","Variables","Lambda")

ggplot(data=plotlasso,aes(x=Lambda,y=`Mean-Squared Error`))+geom_point()+theme_minimal()+
  ylab("log(Lambda)")

coef(lasso_fit)


set.seed(12698282)
n<-5000
x1<-rnorm(n,mean=2,sd=0.5)
x2<-rnorm(n,mean=-1,sd=0.1)
x3<-x1*x2
sigma<-1
error<-rnorm(n,mean=0,sd=sigma)
y<-4+0.9*x1+3*x2+error


database<-cbind.data.frame(y,x1,x2,x3)
nullmodel<-lm(y~1, data=database)
fullmodel<-lm(y~., data=database)
model_step_f <- step(nullmodel,scope=list(lower=nullmodel, upper=fullmodel),direction='forward')
model_step_b <- step(fullmodel,direction='backward')

summary(model_step_f)








# 
# dummylistforward[[1]][4]
# 
# sigma<-c(0.1,0.5,1)
# n<-c(25,100,200,500,5000)
# 
# model_step_b<-matrix(list(), nrow=5,ncol=3)
# model_step_f<-list(list())
# summaryb<-list()
# summaryf<-list()
# 
# for (i in 1:length(n)){
#   for (j in 1:length(sigma)){
#     set.seed(12698282)
#     x1<-rnorm(n[i],mean=2,sd=0.5)
#     x2<-rnorm(n[i],mean=-1,sd=0.1)
#     x3<-x1*x2
#     error<-rnorm(n,mean=0,sd=sigma[j])
#     y<-4+0.9*x1+3*x2+error
#     database<-cbind.data.frame(y,x1,x2,x3)
#     nullmodel<-lm(y~1, data=database)
#     fullmodel<-lm(y~., data=database)
#     model_step_f[i][j] <- step(nullmodel,scope=list(lower=nullmodel, upper=fullmodel),direction='forward')
#     model_step_b[i][j] <- step(fullmodel,direction='backward')
#     summaryb[i][j]<-summary(model_step_b[i][j])
#     summaryf[i][j]<-summary(model_step_f[i][j])
#   }
# }



#sample code for hw2 p3
#monte carlo simulation
n <- 200 #sample size
m <- 100 #number of simulation iterations
#part i) simulate predictors X
x1<-rnorm(n=n, mean=2, sd = 0.4)

x2<-rnorm(n=m, mean=-1, sd = 0.1)

coefl<-c(5,1.2,3)
betaMatrix<-matrix(NA,nrow = 100,ncol = 3)
listMSE<-matrix(NA,nrow = 100,ncol = 1)

#part ii)
for(j in 1:m){
  #simulate the error term m=100 times...
  #generate response vector y with error term per iteration
  error<-rnorm(n=m, mean=0, sd =1)
  y<-coefl[1]+cbind(x1,x2)%*%coefl[-1]+error
  lm.out <- lm(y~.,data=data.frame(y,x1=x1,x2=x2)) #fit linear regression
  betaMatrix[j,] <- lm.out$coefficients
  mysummary <- summary(lm.out)
  listMSE[j] <- mysummary$sigma^2 #get MSE per iteration
}
#part iii) compute MSE bias etc
beta_MSE <- apply(listMSE,2, mean)
beta_mean <- apply(betaMatrix,2,mean)
beta_var <- apply(betaMatrix,2,var)

#Estimationbias 
est_bias<- beta_mean-coefl

