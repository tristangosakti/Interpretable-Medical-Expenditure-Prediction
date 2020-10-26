#Woody Wang (wwang153@stanford.edu) and Tristan Gosakti (tgosakti@stanford.edu), MS&E 226 Project Part 1

set.seed(1)
library(tidyverse)
library(cvTools)
library(GGally)
#install.packages("glmnet", repos = "http://cran.us.r-project.org")
library(glmnet)
#install.packages("InformationValue") 
library(InformationValue)
#install.packages("boot") 
library(boot)

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

ggplot(train, aes(x=med, color='red')) + 
  geom_histogram(bins=500, alpha=0.5)

sum(train$med > 500) / nrow(train) # 500 is the threshold for determining high and low medical expenditures

# ggpairs for correlations
sample_plot_idxs = sample(nrow(train), size = 500) 
sample = train[sample_plot_idxs, ]
ggpairs(sample)

ggplot(train, aes(x=black, color='red')) + 
  geom_histogram(stat="count", alpha=0.5) # dataset has a high percentage of black people

# Look at distributions of medical expenditure with respect to categorical covariates
ggplot(train, aes(x=log(med + 1e-4), fill=sex, color=sex)) + 
  geom_density(bins=500, alpha=0.5)

ggplot(train, aes(x=log(med + 1e-4), fill=child, color=child)) + 
  geom_density(bins=500, alpha=0.5)

ggplot(train, aes(x=log(med + 1e-4), fill=idp, color=idp)) + 
  geom_density(bins=500, alpha=0.5)

ggplot(train, aes(x=log(med + 1e-4), fill=health, color=health)) + 
  geom_density(bins=500, alpha=0.5)

ggplot(train, aes(x=log(med + 1e-4), fill=black, color=black)) + 
  geom_density(bins=500, alpha=0.5)

ggplot(train, aes(x=log(med + 1e-4), fill=physlim, color=physlim)) + 
  geom_density(bins=500, alpha=0.5)

# Look at distributions of medical expenditure w.r.t. continuous covariates
ggplot(train, aes(x=age, y=log(med + 1e-4), color='red')) + 
  geom_point(alpha=0.5)

ggplot(train, aes(x=age, y=med)) + 
  geom_count(alpha=0.5)

ggplot(train, aes(x=ndisease, y=med)) + 
  geom_count(alpha=0.5)

ggplot(train, aes(x=educdec, y=med)) + 
  geom_count(alpha=0.5)

ggplot(train, aes(x=lfam, y=med)) + 
  geom_count(alpha=0.5)

ggplot(train, aes(x=ndisease * age, y=med)) + 
  geom_count(alpha=0.5)

ggplot(train, aes(x=exp(lfam), y=med)) + 
  geom_count(alpha=0.5)

# --------------------------------------- Regression -------------
rmse <- function(model, data) {
  preds = predict(model, data)
  curr_rmse = sqrt(mean((data$med - preds)^2))
  return(curr_rmse)
}

fm = lm(med ~ . - ndisease_by_age, data=train)
cv_fm = cvFit(fm, data=train, y=train$med, K=10, seed=1)
print(rmse(fm, train)) #850.5706
print(cv_fm) #859.9441

fm_1 = lm(med ~ ., data=train)
cv_fm_1 = cvFit(fm_1, data=train, y=train$med, K=10, seed=1)
print(rmse(fm_1, train)) #850.5261
print(cv_fm_1) #860.1419


fm_two_way = lm(med ~ . + .:., data=train)
cv_fm_two_way = cvFit(fm_two_way, data=train, y=train$med, K=10, seed=1)
print(rmse(fm_two_way, train)) #762.6894
print(cv_fm_two_way) #919.1451
plot(train$med, predict(fm_two_way, train))

# Baseline of always predicting mean
fm_mean = lm(med ~ 1, data=train)
cv_fm_mean= cvFit(fm_mean, data=train, y=train$med, K=10, seed=1)
#mean_med = mean(train$med)
#print(mean_med)
print(rmse(fm_mean, train)) #867.9189
print(cv_fm_mean) #868.0392

# L2 model
lambdas <- 10^seq(3, -2, by = -.1)
print(lambdas)
matrix_train <- model.matrix(med ~ ., train)[, -1]
fm_ridge <- glmnet(matrix_train, train$med, alpha = 0, lambda = lambdas, standardize=TRUE)
plot(fm_ridge, xvar = "lambda")

cv_fm_ridge <- cv.glmnet(
  x = matrix_train,
  y = train$med,
  alpha = 0
)
plot(cv_fm_ridge)
opt_lambda = cv_fm_ridge$lambda.min
print(opt_lambda) #607.1292

preds = predict(fm_ridge,s = opt_lambda, matrix_train)
curr_rmse = sqrt(mean((train$med - preds)^2))
print(curr_rmse) # 854.1291, 852.7343
print(sqrt(min(cv_fm_ridge$cvm))) # 861.4073, 858.4869

# L1 model
lambdas <- 10^seq(3, -2, by = -.1)
matrix_train <- model.matrix(med ~ ., train)[, -1]
print(dim(matrix_train))
fm_lasso <- glmnet(matrix_train, train$med, alpha = 1, lambda = lambdas, standardize=TRUE)
plot(fm_lasso, xvar = "lambda")

cv_fm_lasso <- cv.glmnet(
  x = matrix_train,
  y = train$med,
  alpha = 0,
  seed = 1
)
plot(cv_fm_lasso)
opt_lambda = cv_fm_lasso$lambda.min
print(opt_lambda)

print(coef(fm_lasso, s=50))

preds = predict(fm_lasso, s = opt_lambda, matrix_train)
curr_rmse = sqrt(mean((train$med - preds)^2))
print(curr_rmse) # 867.9189
print(sqrt(min(cv_fm_lasso$cvm))) # 858.3144 #TODO: INDEX CORRECTLY FROM LAMBDAS
plot(train$med, preds)

# transform the data
transform_train <- train %>% 
        mutate(fam = exp(lfam)) %>%
        mutate(fam_by_ndisease = fam * ndisease) %>%
        mutate(ndisease_sqr = ndisease^2) %>%
        mutate(age_sqr = age^2)
fm_transform = lm(med ~ ., data=transform_train)
cv_fm_transform = cvFit(fm_transform, data=transform_train, y=transform_train$med, K=10, seed=1)
print(rmse(fm_transform, transform_train))
print(cv_fm_transform)
plot(transform_train$med, predict(fm_transform, transform_train))

# Plot residuals vs observed values
residuals = residuals(fm_transform)
fitted = fitted(fm_transform)
ggplot(data = transform_train) +
  geom_point(mapping = aes(x = fitted, y=residuals))
observed = transform_train$med
ggplot(data = transform_train) +
  geom_point(mapping = aes(x = observed, y=residuals))

# just health baseline
fm_health = lm(med ~ health, data=train)
cv_fm_health = cvFit(fm_health, data=train, y=train$med, K=10, seed=1)
print(rmse(fm_health, train)) # 857.7627
print(cv_fm_health) # 863.859 
plot(train$med, predict(fm_health, train))

# 1) use Lasso to select covariates
# 2) retrain using only the selected covariates
# select physlim, health, age, child, ndisease_by_age (justified by lambda=50 in Lasso)
fm_lasso_refined = lm(med ~ physlim + health + age + child + ndisease_by_age, data=train)
cv_fm_lasso_refined = cvFit(fm_lasso_refined, data=train, y=train$med, K=10, seed=1)
print(rmse(fm_lasso_refined, train)) # 853.2468
print(cv_fm_lasso_refined) # 860.5801 
plot(train$med, predict(fm_lasso_refined, train))
print(fm_lasso_refined$coefficients)

# ----------- Classification ------------------
# Should report precision, recall, and F1 score

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

# Baseline use all covariates
c_train <- train %>% mutate(med = as.numeric(med > 500)) # 500 is predefined threshold
#print(sum(c_train$med ==  1) / nrow(c_train)) # Sanity check 0.06907378
logit_0 <- glm(c_train$med ~ ., data = c_train, family = "binomial")

predicted <- plogis(predict(logit_0, c_train))
optCutOff <- optimalCutoff(c_train$med, predicted)[1]
misClassError(c_train$med, predicted, threshold = optCutOff)
cv_logistic(c_train, K=10, cutoff=optCutOff, model='all_covariates')
plotROC(c_train$med, predicted)
confusionMatrix(c_train$med, predicted, threshold = optCutOff) # columns are actual, rows are pred
curr_recall = sensitivity(c_train$med, predicted, threshold = optCutOff)
print(curr_recall)
curr_precision = precision(c_train$med, predicted, threshold = optCutOff)
print(curr_precision)
curr_F2 = F2(curr_recall, curr_precision)
print(curr_F2)

# Baseline predict mode
logit_mode <- glm(c_train$med ~ 1, data = c_train, family = "binomial")
predicted <- plogis(predict(logit_mode, c_train))
optCutOff = 0.5
misClassError(c_train$med, predicted, threshold = optCutOff)
cv_logistic(c_train, K=10, cutoff=optCutOff, model='mode')
plotROC(c_train$med, predicted)
confusionMatrix(c_train$med, predicted, threshold = optCutOff) # columns are actual, rows are pred
curr_recall = sensitivity(c_train$med, predicted, threshold = optCutOff)
print(curr_recall)
curr_precision = precision(c_train$med, predicted, threshold = optCutOff)
print(curr_precision)
curr_F2 = F2(curr_recall, curr_precision)
print(curr_F2)

# L2 regularized logistic regression
lambdas <- 10^seq(3, -2, by = -.1)
print(lambdas)
matrix_c_train <- model.matrix(med ~ ., c_train)[, -1]
logit_ridge <- glmnet(matrix_c_train, c_train$med, alpha = 0, lambda = lambdas, standardize=TRUE)
plot(logit_ridge, xvar = "lambda")

cv_logit_ridge <- cv.glmnet(
  x = matrix_c_train,
  y = c_train$med,
  alpha = 0
)
plot(cv_logit_ridge)
opt_lambda = cv_logit_ridge$lambda.min
print(opt_lambda)

predicted <- plogis(predict(logit_ridge, s = opt_lambda, matrix_c_train))
optCutOff <- optimalCutoff(c_train$med, predicted)[1] 
optCutOff = 0.55
misClassError(c_train$med, predicted, threshold = optCutOff)
print(min(cv_logit_ridge$cvm)) 
plotROC(c_train$med, predicted)
confusionMatrix(c_train$med, predicted, threshold = optCutOff)
curr_recall = sensitivity(c_train$med, predicted, threshold = optCutOff)
print(curr_recall)
curr_precision = precision(c_train$med, predicted, threshold = optCutOff)
print(curr_precision)
curr_F2 = F2(curr_recall, curr_precision)
print(curr_F2)

# Transform the data
transform_c_train <- c_train %>% 
  mutate(fam = exp(lfam)) %>%
  mutate(fam_by_ndisease = fam * ndisease) %>%
  mutate(ndisease_sqr = ndisease^2) %>%
  mutate(age_sqr = age^2)
logit_transform <- glm(transform_c_train$med ~ ., data = transform_c_train, family = "binomial")
predicted <- plogis(predict(logit_0, transform_c_train))
optCutOff <- optimalCutoff(transform_c_train$med, predicted)[1] 
misClassError(transform_c_train$med, predicted, threshold = optCutOff)
cv_logistic(transform_c_train, K=10, cutoff=optCutOff, model='all_covariates')
plotROC(transform_c_train$med, predicted)
confusionMatrix(transform_c_train$med, predicted, threshold = optCutOff) # columns are actual, rows are pred
curr_recall = sensitivity(transform_c_train$med, predicted, threshold = optCutOff)
print(curr_recall)
curr_precision = precision(transform_c_train$med, predicted, threshold = optCutOff)
print(curr_precision)
curr_F2 = F2(curr_recall, curr_precision)
print(curr_F2)
print(logit_transform$coefficients)

# Just health
logit_health <- glm(c_train$med ~ health, data = c_train, family = "binomial")
predicted <- plogis(predict(logit_health, c_train))
optCutOff <- optimalCutoff(c_train$med, predicted)[1] 
misClassError(c_train$med, predicted, threshold = optCutOff)
cv_logistic(c_train, K=10, cutoff=optCutOff, model='health')
plotROC(c_train$med, predicted)
confusionMatrix(c_train$med, predicted, threshold = optCutOff) # columns are actual, rows are pred
curr_recall = sensitivity(c_train$med, predicted, threshold = optCutOff)
print(curr_recall)
curr_precision = precision(c_train$med, predicted, threshold = optCutOff)
print(curr_precision)
curr_F2 = F2(curr_recall, curr_precision)
print(curr_F2)

# Two way interactions
logit_two_way <- glm(c_train$med ~ . + .:., data = c_train, family = "binomial")
predicted <- plogis(predict(logit_two_way, c_train))
optCutOff <- optimalCutoff(c_train$med, predicted)[1] 
misClassError(c_train$med, predicted, threshold = optCutOff)
cv_logistic(c_train, K=10, cutoff=optCutOff, model='two_way')
plotROC(c_train$med, predicted)
confusionMatrix(c_train$med, predicted, threshold = optCutOff) # columns are actual, rows are pred
curr_recall = sensitivity(c_train$med, predicted, threshold = optCutOff)
print(curr_recall)
curr_precision = precision(c_train$med, predicted, threshold = optCutOff)
print(curr_precision)
curr_F2 = F2(curr_recall, curr_precision)
print(curr_F2)

# L1 regularized logistic regression
lambdas <- 10^seq(3, -2, by = -.1)
print(lambdas)
matrix_c_train <- model.matrix(med ~ ., c_train)[, -1]
logit_lasso <- glmnet(matrix_c_train, c_train$med, alpha = 1, lambda = lambdas, standardize=TRUE)
plot(logit_lasso, xvar = "lambda")

cv_logit_lasso <- cv.glmnet(
  x = matrix_c_train,
  y = c_train$med,
  alpha = 0
)
plot(cv_logit_lasso)
opt_lambda = cv_logit_lasso$lambda.min
print(opt_lambda)

print(coef(logit_lasso, s=0.02))

opt_lambda = 0.01
predicted <- plogis(predict(logit_lasso, s = opt_lambda, matrix_c_train))
optCutOff <- optimalCutoff(c_train$med, predicted)[1] 
optCutOff = 0.55
misClassError(c_train$med, predicted, threshold = optCutOff)
print(min(cv_logit_lasso$cvm)) 
plotROC(c_train$med, predicted)
confusionMatrix(c_train$med, predicted, threshold = optCutOff)
curr_recall = sensitivity(c_train$med, predicted, threshold = optCutOff)
print(curr_recall)
curr_precision = precision(c_train$med, predicted, threshold = optCutOff)
print(curr_precision)
curr_F2 = F2(curr_recall, curr_precision)
print(curr_F2)

# 1) use Lasso to select covariates
# 2) retrain using only the selected covariates
# Select health, age, child, ndisease_by_age

logit_custom <- glm(c_train$med ~ health + age + child + ndisease_by_age, data = c_train, family = "binomial")
predicted <- plogis(predict(logit_custom, c_train))
optCutOff <- optimalCutoff(c_train$med, predicted)[1] 
optCutOff = 0.1
misClassError(c_train$med, predicted, threshold = optCutOff)
cv_logistic(c_train, K=10, cutoff=optCutOff, model='custom')
plotROC(c_train$med, predicted)
confusionMatrix(c_train$med, predicted, threshold = optCutOff) # columns are actual, rows are pred
curr_recall = sensitivity(c_train$med, predicted, threshold = optCutOff)
print(curr_recall)
curr_precision = precision(c_train$med, predicted, threshold = optCutOff)
print(curr_precision)
curr_F2 = F2(curr_recall, curr_precision)
print(curr_F2)


