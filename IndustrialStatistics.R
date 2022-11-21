#The data set OJ from the R package ISLR containing 1070 observations which describe purchases of orange juices (of the type Citrus Hill or Minute Maid)
#Skills used: Pearson’s Chi-squared test, Classification Tree Model, Confusion Tables, Random Forest Classification Model

library(ISLR)
library(rpart)
library(partykit)
library(randomForest)


OJ <- read.csv("IS_OJ.csv", header=TRUE, sep=",", stringsAsFactors = TRUE)
attach(OJ)
set.seed(108)

# A two-way contingency table of categorical nominal variables Purchase and STORE
round(addmargins(prop.table(table(Purchase,STORE)))*100,2)

#H0: The distribution of purchases of two types of orange juice is the same across all the stores (Purchase
and STORE are independent)
#H1: There is a significant difference in the distribution of purchases of two types of orange juice across the
stores (Purchase and STORE are dependent)
chisq.test(table(STORE,Purchase))

# Use the classification tree model with maximum depth level of 2
(class_model_1 <- rpart(Purchase ~., OJ[2:18], control=rpart.control(maxdepth = 2)))
plot(as.party(class_model_1), type="extended")

# The confusion table and the misclassification rate of the training data fit 
fit_1 <- predict(class_model_1, newdata=OJ[2:18], type="class")
(conf_1 <- table(Purchase, fit_1))

cat("Overall misclassification error: ",
ome_1 <- round(100*(conf_1[1,2]+conf_1[2,1])/sum(conf_1),2),"%")

#Use the classification tree model with maximum depth level of 3
(class_model_2 <- rpart(Purchase ~., OJ[2:18], control=rpart.control(maxdepth = 3)))
plot(as.party(class_model_2), type="extended")

# The confusion table and the misclassification rate of the training data fit 
fit_2 <- predict(class_model_2, newdata=OJ[2:18], type="class")
(conf_2 <- table(Purchase, fit_2))

cat("Overall misclassification error: ",
ome_2 <- round(100*(conf_2[1,2]+conf_2[2,1])/sum(conf_2),2),"%")

# Use the random forest classification model with Purchase as the response variable
#The result of a randomForest function call already gives us the corresponding confusion matrix
(class_model_3 <- randomForest(Purchase~., data=OJ[2:18], importance=TRUE))

#In order to determine which model performs best for classifying the Minute Maid orange juice, we compare
the values from the given confusion matrices

# MODEL 1
(acc_1 <- 100*conf_1[2,2]/sum(conf_1[2,]))

# MODEL 2
(acc_2 <- 100*conf_2[2,2]/sum(conf_2[2,]))

# MODEL 3
(acc_3 <- 100*(1-class_model_3$confusion[2,3]))









