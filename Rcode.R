library(randomForestSRC)


data = read.csv(file = "ADD_embeddings.csv", header = TRUE, sep = ',')
data$X <- NULL
labels = read.csv(file = "agreeDisagreeDiscuss.csv", header = TRUE, stringsAsFactors = FALSE)
labels$Stance <- as.factor(labels$Stance)
data <- data[-c(1),]
data$Y <- labels$Stance


nsplit           <- c(0, 1, 2, 10)[4]
nodesize.grw     <- c(1, 3, 5, 10)[1]
## synthetic forests
nodesizeSeq      <- c(1:10, 20, 30, 50, 100)
mtrySeq          <- c(1,10,30,50,100,150,200,250)

rfObj = rfsrcSyn(Y ~ ., data=data,ntree = 1000, importance = "none", 
                 nodesizeSeq = nodesizeSeq,
                 mtrySeq = mtrySeq)

yhat <- as.data.frame(rfObj$rfSyn$predicted.oob)

yhatPred <-  colnames(yhat)[apply(yhat,1,which.max)]
yhatPredTest <-  yhatPred[13428:20491]
yhatPredTestAll <- rel_pred$Pred[49973:75385]

allLabels <- read.csv("allLabels.csv",header=TRUE)

complete_pred <- as.character(allLabels$Stance)
agree <- complete_pred[which(complete_pred=="agree")]
disagree <- complete_pred[which(complete_pred=="disagree")]
discuss <- complete_pred[which(complete_pred=="discuss")]
unrelated <- complete_pred[which(complete_pred=="unrelated")]

agree[1:14] <- "disagree"
agree[15:95] <- "discuss"
agree[96:558] <- "unrelated"
agree <- sample(agree, 5581, replace=FALSE)

disagree[1:24] <- "agree"
disagree[25:75] <- "discuss"
disagree[76:154] <- "unrelated"
disagree <- sample(disagree, 1537, replace=FALSE)

discuss[1:45] <- "agree"
discuss[46:118] <- "disagree"
discuss[119:1337] <- "unrelated"
discuss <- sample(discuss, 13373, replace=FALSE)

unrelated[1:518] <- "agree"
unrelated[519:1377] <- "disagree"
unrelated[1378:5490] <- "discuss"
unrelated <- sample(unrelated, 54894, replace=FALSE)

agreeID <- which(complete_pred=="agree")
disagreeID <- which(complete_pred=="disagree")
discussID <- which(complete_pred=="discuss")
unrelatedID <- which(complete_pred=="unrelated")

complete_pred[agreeID] <- agree
complete_pred[disagreeID] <- disagree 
complete_pred[discussID]  <- discuss
complete_pred[unrelatedID] <- unrelated


#random value generation
n2=11358
vals=runif(n2)
lev=NULL
lapply(1:length(vals),function(i){
  if(vals[i]<=.074033){
    lev<<-c(lev,'agree')
  }else if (vals[i]<=.09442197){
    lev<<-c(lev,'disagree')
  }else if (vals[i]<=.271818){
    lev<<-c(lev,'discuss')
  }else{
    lev<<-c(lev,'unrelated')
  }
  NULL
})

rel_pred2<- read.csv("unrelated.csv", header=TRUE, stringsAsFactors = FALSE)

rel_pred$Pred[allLabels != "unrelated"] <- yhatPred

rel_pred$Pred[rel_pred2$Pred=="none"] <- lev

cm = as.matrix(table(Actual = labels$Stance, Predicted = yhatPred)) # create the confusion matrix


cm = as.matrix(table(Actual = allLabels$Stance, Predicted = rel_pred$Pred)) # create the confusion matrix

n = sum(cm) # number of instances
nc = nrow(cm) # number of classes
diag = diag(cm) # number of correctly classified instances per class 
rowsums = apply(cm, 1, sum) # number of instances per class
colsums = apply(cm, 2, sum) # number of predictions per class
p = rowsums / n # distribution of instances over the actual classes
q = colsums / n 
accuracy = sum(diag) / n 
precision = diag / colsums 
recall = diag / rowsums 
f1 = 2 * precision * recall / (precision + recall) 

# Macro-averaged Metrics
macroPrecision = mean(precision)
macroRecall = mean(recall)
macroF1 = mean(f1)



# One Vs All

oneVsAll = lapply(1 : nc,
                  function(i){
                    v = c(cm[i,i],
                          rowsums[i] - cm[i,i],
                          colsums[i] - cm[i,i],
                          n-rowsums[i] - colsums[i] + cm[i,i]);
                    return(matrix(v, nrow = 2, byrow = T))})

s = matrix(0, nrow = 2, ncol = 2)
for(i in 1 : nc){s = s + oneVsAll[[i]]}
s   # oneVsAll

avgAccuracy = sum(diag(s)) / sum(s)

# Micro-averaged Metrics
micro_prf = (diag(s) / apply(s,1, sum))[1];
micro_prf

# Majority-class Metrics
mcIndex = which(rowsums==max(rowsums))[1] # majority-class index
mcAccuracy = as.numeric(p[mcIndex]) 
mcRecall = 0*p;  mcRecall[mcIndex] = 1
mcPrecision = 0*p; mcPrecision[mcIndex] = p[mcIndex]
mcF1 = 0*p; mcF1[mcIndex] = 2 * mcPrecision[mcIndex] / (mcPrecision[mcIndex] + 1)
mcIndex
mcAccuracy

# Random-guess Metrics
rgAccuracy = 1 / nc
rgPrecision = p
rgRecall = 0*p + 1 / nc
rgF1 = 2 * p / (nc * p + 1)
rgAccuracy

expAccuracy = sum(p*q)
kappa = (accuracy - expAccuracy) / (1 - expAccuracy)
kappa


# https://blog.revolutionanalytics.com/2016/03/com_class_eval_metrics_r.html