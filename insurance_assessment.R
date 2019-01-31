data <- read.csv("C:/Users/채진영/Desktop/prudential_life_insurnce_assessment/train.csv")
test <- read.csv("C:/Users/채진영/Desktop/prudential_life_insurnce_assessment/test.csv")
test_y <- read.csv("C:/Users/채진영/Desktop/prudential_life_insurnce_assessment/sample_submission.csv")

str(data)
str(test)
test$Response <- test_y$Response 
data$train_flag <- 1
test$train_flag <- 0

data <- rbind(data,test)
head(data)

# data check

head(data)
str(data)
summary(data) # Ins_Age, Ht, Wt, BMI normalization  

# NA check
colSums(is.na(data))

data <- data[,colSums(is.na(data))<25000]

# new feature create (Medical_Keyword)

data$keyword <- rowSums(data[,grep("Medical_Keyword",colnames(data))])

data <- data[,-grep("Medical_Keyword",colnames(data))]

table(data$keyword)
quantile(data$keyword, seq(0,1,0.01)) # binding 6 when above 6

data$keyword[data$keyword>6]<-6

table(data$keyword)
quantile(data$keyword, seq(0,1,0.01))

# new feature create (Medical_History)

head(data[,grep("Medical_History",colnames(data))])

for( i in grep("Medical_History",colnames(data))) {
print(colnames(data)[i])
print(table(data[,i]))

}

data$history <- rowSums(is.na(data[,grep("Medical_History",colnames(data))]))

colnames(data)
cor(data[,c(-1,-3,-74,-75,-76)])

data <- data[,-grep("Medical_History",colnames(data))]

str(data)

# Ht, Wt, BMI quantile check

grouping <- function(var,p){
  cut(var, 
      breaks= unique(quantile(var,probs=seq(0,1,by=p), na.rm=T)),
      include.lowest=T, ordered=T) 
}


quantile(data$Ht,seq(0,1,0.25))
quantile(data$Wt)
quantile(data$BMI)


data$HT_group <- grouping(data$Ht, 0.25)
data$WT_group <- grouping(data$Wt, 0.25)
data$BMI_group <- grouping(data$BMI, 0.25)

table(data$HT_group)
table(data$WT_group)
table(data$BMI_group)

str(data)
levels(data$HT_group) <- c(1,2,3,4)
levels(data$WT_group) <- c(1,2,3,4)
levels(data$BMI_group) <- c(1,2,3,4)


data$HT_group <- as.integer(data$HT_group)
data$WT_group <- as.integer(data$WT_group)
data$BMI_group <- as.integer(data$BMI_group)
data[,3]<- as.integer(data[,3])

data <- data[,colnames(data) != c("Ht","Wt","BMI") ]

#############################################he######################

head(data)
str(data)

# test_index <- sample(nrow(data),nrow(data)*0.3)


train <- data[data$train_flag==1,]
test <- data[data$train_flag==0,]

# test <- data[test_index,]
# train <- data[-test_index,]

train$train_flag <- NULL
test$train_flag <- NULL

colnames(train)
colnames(test)





#####################################################################
library(xgboost)
library(caret)   
library(dplyr)
library(e1071)
library(MASS)
library(DAAG)
library(Ckmeans.1d.dp)

colnames(train)
colnames(test)

train_data<- as.matrix(train[,colnames(train)!='Response'])
train_label <- train[,"Response"]-1
train_matrix <- xgb.DMatrix(data = train_data, label = train_label)

test_data<- as.matrix(test[,colnames(test)!='Response'])
test_label <- test[,"Response"]-1
test_matrix <- xgb.DMatrix(data = test_data, label = test_label)


numberOfClasses <- length(unique(train$Response))

xgb_params <- list("objective" = "multi:softprob",
                   "eval_metric" = "mlogloss",
                   "num_class" = numberOfClasses)

nround    <- 50 # number of XGBoost rounds
cv.nfold  <- 3

# Fit cv.nfold * cv.nround XGB models and save OOF predictions

cv_model <- xgb.cv(params = xgb_params,
                   data = train_matrix, 
			 #label = data$Response, 
                   nrounds = nround,
                   nfold = cv.nfold,
                   verbose = FALSE,
                   prediction = TRUE)

OOF_prediction <- data.frame(cv_model$pred) %>%
  mutate(max_prob = max.col(., ties.method = "last"),
         label = data_label)

# prob <- data.frame(cv_model$pred)
# for( i in 1:nrow(cv_model$pred)) {
# prob$max_prob[i] <-which.max(cv_model$pred[i,])
# }
# prob$label <- train_label+1

# confusionMatrix(factor(prob$max_prob),
                factor(prob$label))

# confusion(factor(prob$max_prob),
                factor(prob$label))


confusionMatrix(factor(OOF_prediction$max_prob+1),
                factor(OOF_prediction$label))



bst_model <- xgb.train(params = xgb_params,
                       data = train_matrix,
                       nrounds = nround)

# Predict hold-out test set
test_pred <- predict(bst_model, newdata = test_matrix)
test_prediction <- matrix(test_pred, nrow = numberOfClasses,
                          ncol=length(test_pred)/numberOfClasses) %>%
  t() %>%
  data.frame() %>%
  mutate(label = test_label + 1,
         max_prob = max.col(., "last"))

confusion(factor(test_prediction$max_prob),
                factor(test_prediction$label))

# test_prediction<- matrix(test_pred, nrow=numberOfClasses, 
# ncol=length(test_pred)/numberOfClasses)

# test_prediction<-t(test_prediction)
# test_prediction <- data.frame(test_prediction)



# row.max <- function(x) {
# max_prob <- apply(x,1,max)
# max_label <- which(x==max_prob)
# return ( max_label) 
# }

# for( i in 1:nrow(test_prediction)) {
# test_prediction$max_label[i]<- row.max(test_prediction[i,])
# }

# test_prediction$max_label[test_prediction$max_label==9]<-8

# test_prediction$label <- test_label+1


# table(factor(test_prediction$max_label),
                factor(test_prediction$label))

##################################################################

# feature importance


names <-  colnames(train[,colnames(train)!='Response'])
# compute feature importance matrix
importance_matrix = xgb.importance(feature_names = names, model = bst_model)
head(importance_matrix)


gp = xgb.ggplot.importance(importance_matrix)
print(gp) 














