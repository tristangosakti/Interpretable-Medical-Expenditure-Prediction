#Woody Wang (wwang153@stanford.edu) and Tristan Gosakti (tgosakti@stanford.edu), MS&E 226 Project Part 2. 19-Nov-2020

set.seed(1)
library(tidyverse)
library(cvTools)
library(GGally)
install.packages("glmnet", repos = "http://cran.us.r-project.org")
library(glmnet)
#install.packages("InformationValue") 
library(InformationValue)
install.packages("boot") 
library(boot)
#install.packages("sgof")
#library("sgof") don't need this import, use p.adjust

data = read.csv("/Users/tristangosakti/Desktop/ms&e226/MedExp.csv", sep=',')
data$idp = factor(data$idp)
data$physlim = factor(data$physlim)
data$health = factor(data$health)
data$sex = factor(data$sex)

data <- data %>% mutate(ndisease_by_age = ndisease * age)

data$child = factor(data$child)
data$black = factor(data$black)
data = select(data, -X)


# Hold out test set
idxs = sample(nrow(data), size = 4459) # 80% of the total dataset
train = data[idxs, ] # 4459 total size of train set
test = data[-idxs, ] # 1115 total size of test set


#
rmse <- function(model, data) {
  preds = predict(model, data)
  curr_rmse = sqrt(mean((data$med - preds)^2))
  return(curr_rmse)
}

# 1) use Lasso to select covariates
# 2) retrain using only the selected covariates
# select physlim, health, age, child, ndisease_by_age (justified by lambda=50 in Lasso)
fm_lasso_refined = lm(med ~ physlim + health + age + child + ndisease_by_age, data=train)
cv_fm_lasso_refined = cvFit(fm_lasso_refined, data=train, y=train$med, K=10, seed=1)
print(rmse(fm_lasso_refined, train)) # 853.2468
print(cv_fm_lasso_refined) # 860.5801 
plot(train$med, predict(fm_lasso_refined, train))
print(fm_lasso_refined$coefficients)

########## TEST REGRESSION ##########
print(rmse(fm_lasso_refined, test)) #444.1637 (?) #with best model
head(test)
head(train)

fm_two_way = lm(med ~ . + .:., data=train)
cv_fm_two_way = cvFit(fm_two_way, data=train, y=train$med, K=10, seed=1)
print(rmse(fm_two_way, train)) #762.6894
print(cv_fm_two_way) #919.1451
print(rmse(fm_two_way, test)) #639.534
##################################################

########## TEST CLASSIFICATION (PART 1) ##########

F2 <- function(recall, precision) {
  f_value = (5) * precision * recall / ((4 * precision) + recall)
  return(f_value)
}

cv_logistic <- function(data, K, cutoff, model) {
  #Create 10 equally size folds
  folds <- cut(seq(1,nrow(data)),breaks=K,labels=FALSE)
  
  #Perform 10 fold cross validation
  curr_missClassError = 0.0
  for(i in 1:K){
    #Segment your data by fold using the which() function 
    testIndexes <- which(folds==i,arr.ind=TRUE)
    testData <- data[testIndexes, ]
    trainData <- data[-testIndexes, ]
    #Use test and train data partitions however you desire...
    logit_curr = NULL
    if (model == 'all_covariates') {
      logit_curr <- glm(trainData$med ~ ., data = trainData, family = "binomial")
    }
    else if (model == 'mode') {
      logit_curr <- glm(trainData$med ~ 1, data = trainData, family = "binomial")
    }
    else if (model == 'health') {
      logit_curr <- glm(trainData$med ~ health, data = trainData, family = "binomial")
    }
    else if (model == 'two_way') {
      logit_curr <- glm(trainData$med ~ . + .:., data = trainData, family = "binomial")
    }
    else if (model == 'custom') {
      logit_curr <- glm(trainData$med ~ health + age + child + ndisease_by_age, data = trainData, family = "binomial")
    }
    predicted <- plogis(predict(logit_curr, testData))
    curr_missClassError = curr_missClassError + misClassError(testData$med, predicted, threshold = cutoff)
  }
  curr_missClassError = curr_missClassError / K
  return(curr_missClassError)
}

c_train <- train %>% mutate(med = as.numeric(med > 500)) # 500 is predefined threshold
c_test <- test %>% mutate(med = as.numeric(med > 500)) # 500 is predefined threshold


# 1) use Lasso to select covariates
# 2) retrain using only the selected covariates
# Select health, age, child, ndisease_by_age

  #for train
logit_custom <- glm(c_train$med ~ health + age + child + ndisease_by_age, data = c_train, family = "binomial")
predicted <- plogis(predict(logit_custom, c_train))
optCutOff <- optimalCutoff(c_train$med, predicted)[1] 
optCutOff = 0.1
1-misClassError(c_train$med, predicted, threshold = optCutOff) #0.7975
1-cv_logistic(c_train, K=10, cutoff=optCutOff, model='custom') #0.79526
plotROC(c_train$med, predicted)
confusionMatrix(c_train$med, predicted, threshold = optCutOff) # columns are actual, rows are pred
curr_recall = sensitivity(c_train$med, predicted, threshold = optCutOff)
print(curr_recall) #0.4058442
curr_precision = precision(c_train$med, predicted, threshold = optCutOff)
print(curr_precision) #0.147929
curr_F2 = F2(curr_recall, curr_precision)
print(curr_F2)
print(logit_custom$coefficients)

  #for test
predicted <- plogis(predict(logit_custom, c_test))
1-misClassError(c_test$med, predicted, threshold = optCutOff) #0.7767

confusionMatrix(c_test$med, predicted, threshold = optCutOff) # columns are actual, rows are pred
curr_recall = sensitivity(c_test$med, predicted, threshold = optCutOff)
print(curr_recall) #0.3414634
curr_precision = precision(c_test$med, predicted, threshold = optCutOff)
print(curr_precision) # 0.1255605
curr_F2 = F2(curr_recall, curr_precision) 
print(curr_F2) #0.25

##################################################

########EXPLORING TRAIN VS TEST########
print(max(train$med)) #39182.02
print(max(test$med)) #5996.954
print(mean(train$med)) #173.009
print(mean(test$med)) #156.5901

print(sum(train$med > max(test$med))) #10

ggplot(train, aes(x=med)) + 
  geom_density(bins=500, alpha=0.5, color='red', fill="red")

ggplot(test, aes(x=med)) + 
  geom_density(bins=500, alpha=0.5, color='blue', fill="blue")

########################################################



#################### INFERENCE INFERENCE INFERENCE WEEEOOO ########################

# 1) use Lasso to select covariates
# 2) retrain using only the selected covariates
# select physlim, health, age, child, ndisease_by_age (justified by lambda=50 in Lasso)

fm_lasso_refined = lm(med ~ physlim + health + age + child + ndisease_by_age, data=train)
print(summary(fm_lasso_refined))

fm_lasso_refined_test = lm(med ~ physlim + health + age + child + ndisease_by_age, data=test)
print(summary(fm_lasso_refined_test))


### BOOTSTRAP ###

coef.boot = function(data, indices) {
  #fm = lm(data = data[indices,], Y ~ 1 + X)
  
  fm_lasso_refined = lm(med ~ physlim + health + age + child + ndisease_by_age, data=data[indices,])
  
  return(coef(fm_lasso_refined))
}

boot.out = boot(train, coef.boot, 10000)
print(boot.out)

for(ind in 1:8){
  if (ind == 2){
    plot(boot.out, index=ind)
  }
  cat("ind: ", ind, '\n')
  print(boot.ci(boot.out, type="norm", index=ind))
  
}
###

### RECOMMENDED 1 ####
print(".")
fm_1 = lm(med ~ ., data=train)
print(summary(fm_1))

####

## RECOMMENDED BH ##

p_vals = (summary(fm_lasso_refined))$coefficients[,4]
cat("p_vals: ", p_vals)
cat("bonferroni: ", p.adjust(p=p_vals, method="bonferroni"))

cat("BH: ", p.adjust(p=p_vals, method="BH"))
################################################################################
