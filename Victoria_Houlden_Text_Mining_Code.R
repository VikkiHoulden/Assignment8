#Text Mining Coursework

####################################################################
###         Pre processing for obtaining bag of words            ###
####################################################################
#Load necessary libraries
library(randomForest)
library(NLP)
library(e1071)
library(caret)
library(tm)
library(httr)
library(XML)
library(SnowballC)
library(openNLP)
library(ggplot2)
library(reshape2)
library(koRpus)
library(lda)
library(openNLPmodels.en)
library(NLP)
library(topicmodels)
#Read in dataset and remove non standard symbols to enable analysis
reuters <- read.csv("~/Desktop/reuters.csv")
reuters$doc.text=iconv(reuters$doc.text, "UTF-8", "UTF-8",sub='') 
#Convert to corpus to allow the use of text mining tools
docs=Corpus(VectorSource(reuters$doc.text))
copy=docs
nostop=copy
#Remove excess whitespace and numbers from Corpus and initially remove some stopwords
nospace=tm_map(docs,stripWhitespace)
nonumbers=tm_map(nospace,removeNumbers)
swords=stopwords('english')
for(i in 1:11341){nostop[[i]]=removeWords(nonumbers[[i]],swords)}
#Loop over all documents, tokenizing and lemmatizing words using Treetag tool
for(j in 1:11341)
{
  temp=nostop[[j]][1]
  temp=gsub("-"," ",temp)
  if(temp==""){}else{
    tempaslist=paste(temp, collapse='' )
    #Call treetagger tool for each word, putting in root form
    temptagged=treetag(tempaslist,treetagger="manual", format="obj",TT.tknz=FALSE , lang="en",TT.options=list(path="/Users/Vikki/Desktop/test", preset="en"))
    templemmatised=taggedText(temptagged)[3][1]
    for(k in 1:dim(taggedText(temptagged)[3])[1])
    {
      if(taggedText(temptagged)[3][k,1]=='<unknown>')
      {
        templemmatised[k,1]=taggedText(temptagged)[1][k,1]
      }
      else
      {
        templemmatised[k,1]=taggedText(temptagged)[3][k,1]
      }
    }
    tempcorpus=Corpus(VectorSource(templemmatised))
    #renove any punctuation and newly created whitespace
    nopunctuation=tm_map(tempcorpus,removePunctuation)
    nopunctuationnospace=tm_map(nopunctuation,stripWhitespace)
    copy[[j]]=nopunctuationnospace[[1]]}
}
#Put all words in lower case so they will be recognised together
lowerc=tm_map(copy,tolower)
#Any newly lower cased stop words will now be recognised and removed, including the word 'Reuter'
nostop3=tm_map(lowerc, removeWords, c('reuter',stopwords("english"))) 
nospace2 <- tm_map(nostop3, removespace);
allwords=tm_map(nospace2, PlainTextDocument)
threshold=10
#Document term matrix created for all words which have a frequency higher than 10
dtm=DocumentTermMatrix(allwords, control=list(bounds = list(global = c(threshold,Inf))))
rowTotals <- apply(dtm , 1, sum)
#Any documents which dont have any words from this thresholded term list are removed.
reuters.nonzero=reuters[rowTotals> 0, ]
dtmnonzero   <- dtm[rowTotals> 0, ]
#Convert document term matrix to standard matrix.
wordMatrix = as.data.frame(as.matrix(dtm))
####################################################################
### Sort unigrms by chi squared value to find strongest features ###
####################################################################
chimat=as.data.frame(matrix(0, ncol = dim(wordMatrix)[2], nrow = 118))
#First frequency for each word in each Topic Tag Class is found
for(i in 1:dim(wordMatrix)[2])
{
  docind=which(wordMatrix[,i]!=0)
  for(j in docind)
  {
    classind=which(reuters[j,4:121]!=0)
    for(k in classind)
    {
      chimat[k,i]=chimat[k,i]+1
    }
  }
}
colnames(chimat) <- colnames(wordMatrix)
rownames(chimat) <- colnames(reuters)[4:121]
rowsum=array(0,dim=c(dim(chimat)[1],1))
colsum=array(0,dim=c(dim(chimat)[2],1))
#From these, pairwise chi-squared values can be calculated for ranking features for each Topic
for(i in 1:dim(chimat)[2])
{
  colsum[i]=sum(chimat[,i])
}
for(j in 1:dim(chimat)[1])  
{
  rowsum[j]=sum(chimat[j,])
}
chisqmat=chimat
for(i in 1:dim(chimat)[2])
{
  for(j in 1:dim(chimat)[1])
  {
    N11=chimat[j,i]
    N10=colsum[i]-N11
    N00=N-rowsum[j]-colsum[i]+N11
    N01=rowsum[j]-N11
    chisqmat[j,i]=((N11+N10+N01+N00)*((N11*N00)-(N01*N10))^2)/((N11+N01)*(N11+N10)*(N10+N00)*(N01+N00))
  }
}
#For each Topic Tag, top 10 unigram features are chosen
top10=array(dim=c(dim(chisqmat)[1],10))
for(i in 1:dim(chisqmat)[1])
{
  inorder=sort(chisqmat[i,],decreasing = TRUE)
  top10[i,]=colnames(inorder[1:10])
}
featlist=unique(c(top10))
#Also topic models are explored as features, calling LDA and relative probabilities for each document calculated.
fulltopicmodel=LDA(dtmnonzero,50,method="VEM",control=NULL,model=NULL)
probs=posterior(fulltopicmodel)$topics
#Any words not included in featlist (list of features) removed
for(i in 1:length(featlist))
{
  wordind[i]=which(colnames(wordMatrix)==featlist[i])
}
featmatrix=wordMatrix[,wordind]
####################################################################
###  Using Topic Model and Unigrams as features for classifying  ###
####################################################################
#Only consider most populous Topic Tags as classes for classification as asked
reuters.selectedtags=reuters.nonzero[,c(1,2,3,34,4,66,41,28,115,50,99,117,21,122,123)]
rsum=array(0,dim=c(1,10412))
rsum2=array(0,dim=c(1,10412))
for(k in 1:10412)
{
  rsum[k]=sum(reuters.selectedtags[k,4:13])
  rsum2[k]=sum(featmatrix[k,1:799])
}
#Remove any documents with multiple classes or no features
singletags=which(rsum==1)
nofeatures=which(rsum2==0)
singletags_nonzero=singletags [! singletags %in% nofeatures]
reuters.selectedtags.nonzero=reuters.selectedtags[singletags_nonzero,]
classtag=array(0,dim=c(7793,1))
for(l in 1:7793)
{
  classtag[l]=colnames(reuters.selectedtags.nonzero)[3+which(reuters.selectedtags.nonzero[l,4:13]==1)]
}
#Create a data frame which consists of features and assigned class tag
classtag=as.data.frame(classtag)
featmatrix2=featmatrix[singletags_nonzero,]
probs2=probs[singletags_nonzero,]
unigrammatrix=as.data.frame(array(0,dim=c(dim(featmatrix2)[1],dim(featmatrix2)[2]+dim(classtag)[2])))
unigrammatrix[,1]=classtag
unigrammatrix[,2:800]=featmatrix2
topicmodelmatrix=as.data.frame(array(0,dim=c(dim(probs2)[1],dim(probs2)[2]+dim(classtag)[2])))
topicmodelmatrix[,1]=classtag
topicmodelmatrix[,2:51]=probs2
for(l in 1:7793)
{
  unigrammatrix[l,2:800]=unigrammatrix[l,2:800]/sum(unigrammatrix[l,2:800])
}
colnames(unigrammatrix) <- c('Class',colnames(featmatrix))
colnames(topicmodelmatrix) <- c('Class',colnames(probs2))
#Initially split data into training and test sets
testset_unigram=unigrammatrix[which(reuters.selectedtags.nonzero$purpose=='test'),]
trainingset_unigram=unigrammatrix[which(reuters.selectedtags.nonzero$purpose=='train'),]
testset_topicmodel=topicmodelmatrix[which(reuters.selectedtags.nonzero$purpose=='test'),]
trainingset_topicmodel=topicmodelmatrix[which(reuters.selectedtags.nonzero$purpose=='train'),
xtrain_unigram<-subset(trainingset_unigram, select = -Class)
ytrain_unigram<-trainingset_unigram$Class
xtest_unigram<-subset(testset_unigram, select = -Class)
ytest_unigram<-testset_unigram$Class
#Train Naive Bayes and Random Forest models using both unigram and Topic Model features on training data
#Generate confusion matrices for calculation of accuracy, precision and recall to compare models
forest_unigram_train_model=randomForest(xtrain_unigram,ytrain_unigram)
forest_unigram_train_pred <- predict(forest_unigram_train_model, xtrain_unigram)
forest_unigram_train_table=table(forest_unigram_train_pred, ytrain_unigram)
forest_unigram_test_pred <- predict(forest_unigram_train_model, xtest_unigram)
forest_unigram_test_table=table(forest_unigram_test_pred, ytest_unigram)
bayes_unigram_train_model=naiveBayes(xtrain_unigram,ytrain_unigram)
bayes_unigram_train_pred <- predict(bayes_unigram_train_model, xtrain_unigram)
bayes_unigram_train_table=table(bayes_unigram_train_pred, ytrain_unigram)
bayes_unigram_test_pred <- predict(bayes_unigram_train_model, xtest_unigram)
bayes_unigram_test_table=table(bayes_unigram_test_pred, ytest_unigram)
xtrain_topicmodel<-subset(trainingset_topicmodel, select = -Class)
ytrain_topicmodel<-trainingset_topicmodel$Class
xtest_topicmodel<-subset(testset_topicmodel, select = -Class)
ytest_topicmodel<-testset_topicmodel$Class
forest_topicmodel_train_model=randomForest(xtrain_topicmodel,ytrain_topicmodel)
forest_topicmodel_train_pred <- predict(forest_topicmodel_train_model, xtrain_topicmodel)
forest_topicmodel_train_table=table(forest_topicmodel_train_pred, ytrain_topicmodel)
forest_topicmodel_test_pred <- predict(forest_topicmodel_train_model, xtest_topicmodel)
forest_topicmodel_test_table=table(forest_topicmodel_test_pred, ytest_topicmodel)
bayes_topicmodel_train_model=naiveBayes(xtrain_topicmodel,ytrain_topicmodel)
bayes_topicmodel_train_pred <- predict(bayes_topicmodel_train_model, xtrain_topicmodel)
bayes_topicmodel_train_table=table(bayes_topicmodel_train_pred, ytrain_topicmodel)
bayes_topicmodel_test_pred <- predict(bayes_topicmodel_train_model, xtest_topicmodel)
bayes_topicmodel_test_table=table(bayes_topicmodel_test_pred, ytest_topicmodel)
no_levels=nlevels(ytrain)
#Calculate accuracies from confusion matrices
acc_bayes_topicmodel_train=confusionMatrix(bayes_topicmodel_train_table)
Accuracy_bayes_topicmodel_train=acc_bayes_topicmodel_train$overall[1]
acc_forest_topicmodel_train=confusionMatrix(forest_topicmodel_train_table)
Accuracy_forest_topicmodel_train=acc_forest_topicmodel_train$overall[1]
acc_bayes_unigram_train=confusionMatrix(bayes_unigram_train_table)
Accuracy_bayes_unigram_train=acc_bayes_unigram_train$overall[1]
acc_forest_unigram_train=confusionMatrix(forest_unigram_train_table)
Accuracy_forest_unigram_train=acc_forest_unigram_train$overall[1]
#Initialise precision and recall variables
precision_forest_unigram_train=array(0,dim=c(no_levels,1))
precision_bayes_unigram_train=array(0,dim=c(no_levels,1))
precision_forest_topicmodel_train=array(0,dim=c(no_levels,1))
precision_bayes_topicmodel_train=array(0,dim=c(no_levels,1))
recall_forest_unigram_train=array(0,dim=c(no_levels,1))
recall_bayes_unigram_train=array(0,dim=c(no_levels,1))
recall_forest_topicmodel_train=array(0,dim=c(no_levels,1))
recall_bayes_topicmodel_train=array(0,dim=c(no_levels,1))
#Calculate individual precisions and recalls for each class
for(i in 1:no_levels)
{
  precision_forest_unigram_train[i]=forest_unigram_train_table[i,i]/sum(forest_unigram_train_table[i,])
  recall_forest_unigram_train[i]=forest_unigram_train_table[i,i]/sum(forest_unigram_train_table[,i])
  precision_bayes_unigram_train[i]=bayes_unigram_train_table[i,i]/sum(bayes_unigram_train_table[i,])
  recall_bayes_unigram_train[i]=bayes_unigram_train_table[i,i]/sum(bayes_unigram_train_table[,i])
  precision_forest_topicmodel_train[i]=forest_topicmodel_train_table[i,i]/sum(forest_topicmodel_train_table[i,])
  recall_forest_topicmodel_train[i]=forest_topicmodel_train_table[i,i]/sum(forest_topicmodel_train_table[,i])
  precision_bayes_topicmodel_train[i]=bayes_topicmodel_train_table[i,i]/sum(bayes_topicmodel_train_table[i,])
  recall_bayes_topicmodel_train[i]=bayes_topicmodel_train_table[i,i]/sum(bayes_topicmodel_train_table[,i])
}
#Calculate macro and micro measures of precision and recall
precision_macro_forest_unigram_train=(1/no_levels)*sum(precision_forest_unigram_train,na.rm=TRUE)
precision_macro_bayes_unigram_train=(1/no_levels)*sum(precision_bayes_unigram_train,na.rm=TRUE)
recall_macro_forest_unigram_train=(1/no_levels)*sum(recall_forest_unigram_train,na.rm=TRUE)
recall_macro_bayes_unigram_train=(1/no_levels)*sum(recall_bayes_unigram_train,na.rm=TRUE)
precision_macro_forest_topicmodel_train=(1/no_levels)*sum(precision_forest_topicmodel_train,na.rm=TRUE)
precision_macro_bayes_topicmodel_train=(1/no_levels)*sum(precision_bayes_topicmodel_train,na.rm=TRUE)
recall_macro_forest_topicmodel_train=(1/no_levels)*sum(recall_forest_topicmodel_train,na.rm=TRUE)
recall_macro_bayes_topicmodel_train=(1/no_levels)*sum(recall_bayes_topicmodel_train,na.rm=TRUE)
recall_micro_bayes_topicmodel_train=sum(diag(bayes_topicmodel_train_table))/sum(bayes_topicmodel_train_table,na.rm=TRUE)
precision_micro_bayes_topicmodel_train=sum(diag(bayes_topicmodel_train_table))/sum(bayes_topicmodel_train_table,na.rm=TRUE)
recall_micro_forest_topicmodel_train=sum(diag(forest_topicmodel_train_table))/sum(forest_topicmodel_train_table,na.rm=TRUE)
precision_micro_forest_topicmodel_train=sum(diag(forest_topicmodel_train_table))/sum(forest_topicmodel_train_table,na.rm=TRUE)
recall_micro_bayes_unigram_train=sum(diag(bayes_unigram_train_table))/sum(bayes_unigram_train_table,na.rm=TRUE)
precision_micro_bayes_unigram_train=sum(diag(bayes_unigram_train_table))/sum(bayes_unigram_train_table,na.rm=TRUE)
recall_micro_forest_unigram_train=sum(diag(forest_unigram_train_table))/sum(forest_unigram_train_table,na.rm=TRUE)
precision_micro_forest_unigram_train=sum(diag(forest_unigram_train_table))/sum(forest_unigram_train_table,na.rm=TRUE)
#Random Forest performs best from above measures, so use for test set too
acc_forest_topicmodel_test=confusionMatrix(forest_topicmodel_test_table)
Accuracy_forest_topicmodel_test=acc_forest_topicmodel_test$overall[1]
acc_forest_unigram_test=confusionMatrix(forest_unigram_test_table)
Accuracy_forest_unigram_test=acc_forest_unigram_test$overall[1]
recall_forest_topicmodel_test=array(0,dim=c(no_levels,1))
precision_forest_topicmodel_test=array(0,dim=c(no_levels,1))
recall_forest_unigram_test=array(0,dim=c(no_levels,1))
precision_forest_unigram_test=array(0,dim=c(no_levels,1))
for(i in 1:no_levels)
{
  precision_forest_unigram_test[i]=forest_unigram_test_table[i,i]/sum(forest_unigram_test_table[i,])
  recall_forest_unigram_test[i]=forest_unigram_test_table[i,i]/sum(forest_unigram_test_table[,i])
  precision_forest_topicmodel_test[i]=forest_topicmodel_test_table[i,i]/sum(forest_topicmodel_test_table[i,])
  recall_forest_topicmodel_test[i]=forest_topicmodel_test_table[i,i]/sum(forest_topicmodel_test_table[,i])
}
#Calculate recall and precision for test data
recall_macro_forest_topicmodel_test=(1/no_levels)*sum(recall_forest_topicmodel_test,na.rm=TRUE)
recall_macro_forest_unigram_test=(1/no_levels)*sum(recall_forest_unigram_test,na.rm=TRUE)
precision_macro_forest_topicmodel_test=(1/no_levels)*sum(precision_forest_topicmodel_test,na.rm=TRUE)
precision_macro_forest_unigram_test=(1/no_levels)*sum(precision_forest_unigram_test,na.rm=TRUE)
#Also perform K-fold
permuteddata_unigram <- unigrammatrix[sample(nrow(unigrammatrix)),]
permuteddata_topicmodel<- topicmodelmatrix[sample(nrow(topicmodelmatrix)),]
k=10
no_entries=dim(unigrammatrix)[1]
no_per_subset=no_entries/k;
precision_forest_unigram=array(0,dim=c(no_levels,k))
recall_forest_unigram=array(0,dim=c(no_levels,k))
precision_bayes_unigram=array(0,dim=c(no_levels,k))
recall_bayes_unigram=array(0,dim=c(no_levels,k))
precision_forest_topicmodel=array(0,dim=c(no_levels,k))
recall_forest_topicmodel=array(0,dim=c(no_levels,k))
precision_bayes_topicmodel=array(0,dim=c(no_levels,k))
recall_bayes_topicmodel=array(0,dim=c(no_levels,k))
Accuracy_forest_topicmodel=array(0,dim=c(1,k))
Accuracy_bayes_topicmodel=array(0,dim=c(1,k))
Accuracy_forest_unigram=array(0,dim=c(1,k))
Accuracy_bayes_unigram=array(0,dim=c(1,k))
for(j in 1:k)
{
  temp=((j-1)*no_per_subset)+1:no_per_subset
  test_unigram=permuteddata_unigram[temp,]
  test_topicmodel=permuteddata_topicmodel[temp,]
  training_unigram=permuteddata_unigram[-c(as.numeric(rownames(test_unigram))),]
  training_topicmodel=permuteddata_topicmodel[-c(as.numeric(rownames(test_topicmodel))),]
  xtraining_unigram <- subset(training_unigram, select = -Class)
  ytraining_unigram <- training_unigram$Class
  xtraining_topicmodel <- subset(training_topicmodel, select = -Class)
  ytraining_topicmodel <- training_topicmodel$Class
  xtest_unigram<-subset(test_unigram,select = -Class)
  ytest_unigram<-test_unigram$Class
  xtest_topicmodel<-subset(test_topicmodel,select = -Class)
  ytest_topicmodel<-test_topicmodel$Class
  model_forest_unigram=randomForest(xtraining_unigram,droplevels(ytraining_unigram))
  model_forest_topicmodel=randomForest(xtraining_topicmodel,droplevels(ytraining_topicmodel))
  model_bayes_unigram=naiveBayes(xtraining_unigram,ytraining_unigram)
  model_bayes_topicmodel=naiveBayes(xtraining_topicmodel,ytraining_topicmodel)
  pred_forest_unigram <- predict(model_forest_unigram, xtest_unigram)
  pred_forest_topicmodel <- predict(model_forest_topicmodel, xtest_topicmodel)
  pred_bayes_unigram <- predict(model_bayes_unigram, xtest_unigram)
  pred_bayes_topicmodel <- predict(model_bayes_topicmodel, xtest_topicmodel)
  table_forest_unigram=table(pred_forest_unigram, ytest_unigram)
  table_bayes_unigram=table(pred_bayes_unigram,ytest_unigram)
  table_forest_topicmodel=table(pred_forest_topicmodel, ytest_topicmodel)
  table_bayes_topicmodel=table(pred_bayes_topicmodel,ytest_topicmodel)
  acc_forest_topicmodel=confusionMatrix(table_forest_topicmodel)
  Accuracy_forest_topicmodel[j]=acc_forest_topicmodel$overall[1]
  acc_forest_unigram=confusionMatrix(table_forest_unigram)
  Accuracy_forest_unigram[j]=acc_forest_unigram$overall[1]
  acc_bayes_topicmodel=confusionMatrix(table_bayes_topicmodel)
  Accuracy_bayes_topicmodel[j]=acc_bayes_topicmodel$overall[1]
  acc_bayes_unigram=confusionMatrix(table_bayes_unigram)
  Accuracy_bayes_unigram[j]=acc_bayes_unigram$overall[1] 
  for(l in 1:no_levels-1)
  {
    precision_forest_topicmodel[l,j]=table_forest_topicmodel[l,l]/sum(table_forest_topicmodel[l,])
    recall_forest_topicmodel[l,j]=table_forest_topicmodel[l,l]/sum(table_forest_topicmodel[,l])
    precision_bayes_topicmodel[l,j]=table_bayes_topicmodel[l,l]/sum(table_bayes_topicmodel[l,])
    recall_bayes_topicmodel[l,j]=table_bayes_topicmodel[l,l]/sum(table_bayes_topicmodel[,l])
    precision_forest_unigram[l,j]=table_forest_unigram[l,l]/sum(table_forest_unigram[l,])
    recall_forest_unigram[l,j]=table_forest_unigram[l,l]/sum(table_forest_unigram[,l])
    precision_bayes_unigram[l,j]=table_bayes_unigram[l,l]/sum(table_bayes_unigram[l,])
    recall_bayes_unigram[l,j]=table_bayes_unigram[l,l]/sum(table_bayes_unigram[,l])
  }
}
#Calculate average precision, recall and accuracy across k folds
final_precision_forest_topicmodel=mean(precision_forest_topicmodel)
final_precision_bayes_topicmodel=mean(precision_bayes_topicmodel)
final_precision_forest_unigram=mean(precision_forest_unigram)
final_precision_bayes_unigram=mean(precision_bayes_unigram)
final_recall_forest_topicmodel=mean(recall_forest_topicmodel)
final_recall_bayes_topicmodel=mean(recall_bayes_topicmodel)
final_recall_forest_unigram=mean(recall_forest_unigram)
final_recall_bayes_unigram=mean(recall_bayes_unigram)
final_accuracy_forest_topicmodel=mean(Accuracy_forest_topicmodel)
final_accuracy_bayes_topicmodel=mean(Accuracy_bayes_topicmodel)
final_accuracy_forest_unigram=mean(Accuracy_forest_unigram)
final_accuracy_bayes_unigram=mean(Accuracy_bayes_unigram)
####################################################################
###         Clustering data using Topic Model Features           ###
####################################################################
####Initially use K-means clustering with 60 clusters
fit=kmeans(topicmodelmatrix[,2:51],60)
allocatedcluster=fit$cluster
clustmat=as.data.frame(array(0,dim=c(60,9)))
colnames(clustmat)=c('topic.acq','topic.crude','topic.earn','topic.grain','topic.trade','topic.interest','topic.ship',"topic.money.fx",'topic.wheat')
colnames(clustmat)    
#Create freqeuncy table for each cluster and class tag, to check performance
for(j in 1:60)
{
  for(k in 1:length(which(allocatedcluster==j)))
  {
    if(classtag[which(allocatedcluster==j),][k]=='topic.acq')
    {
      clustmat[j,1]=clustmat[j,1]+1
    }
    else if(classtag[which(allocatedcluster==j),][k]=='topic.crude')
    {
      clustmat[j,2]=clustmat[j,2]+1
    }
    else if(classtag[which(allocatedcluster==j),][k]=='topic.earn')
    {
      clustmat[j,3]=clustmat[j,3]+1
    }
    else if(classtag[which(allocatedcluster==j),][k]=='topic.grain')
    {
      clustmat[j,4]=clustmat[j,4]+1
    }
    else if(classtag[which(allocatedcluster==j),][k]=='topic.trade')
    {
      clustmat[j,5]=clustmat[j,5]+1
    }
    else if(classtag[which(allocatedcluster==j),][k]=='topic.interest')
    {
      clustmat[j,6]=clustmat[j,6]+1
    }
    else if(classtag[which(allocatedcluster==j),][k]=='topic.ship')
    {
      clustmat[j,7]=clustmat[j,7]+1  
    }
    else if(classtag[which(allocatedcluster==j),][k]=='topic.money.fx')
    {
      clustmat[j,8]=clustmat[j,8]+1
    }
    else if(classtag[which(allocatedcluster==j),][k]=='topic.crude')
    {
      clustmat[j,2]=clustmat[j,2]+1
    }
    else if(classtag[which(allocatedcluster==j),][k]=='topic.wheat')
    {
      clustmat[j,9]=clustmat[j,9]+1
    }
  }
}
#Normalise to show the percentage of each cluster for each class tag
normclust=clustmat
for(l in 1:60)
{
  rsumclust=sum(clustmat[l,])
  normclust[l,]=clustmat[l,]/rsumclust
}
###### Also try clustering, using hclust, generate Dendogram, too crowded to analyse
fithclust=hclust(hdist,method="complete")
plot(fithclust)
###### Use DBSCAN as the last clustering approach
fitdb <- dbscan(topicmodelmatrix[,2:51],MinPts = 10,eps=0.1)
allocatedclusterdb=fitdb$cluster
clustmatdb=as.data.frame(array(0,dim=c(19,9)))
colnames(clustmatdb)=c('topic.acq','topic.crude','topic.earn','topic.grain','topic.trade','topic.interest','topic.ship',"topic.money.fx",'topic.wheat')
colnames(clustmatdb)
# Again calculate frequencies for each class and cluster
for(j in 1:19)
{
  for(k in 1:length(which(allocatedclusterdb==j)))
  {
    if(classtag[which(allocatedclusterdb==j),][k]=='topic.acq')
    {
      clustmatdb[j,1]=clustmatdb[j,1]+1
    }
    else if(classtag[which(allocatedclusterdb==j),][k]=='topic.crude')
    {
      clustmatdb[j,2]=clustmatdb[j,2]+1
    }
    else if(classtag[which(allocatedclusterdb==j),][k]=='topic.earn')
    {
      clustmatdb[j,3]=clustmatdb[j,3]+1
    }
    else if(classtag[which(allocatedclusterdb==j),][k]=='topic.grain')
    {
      clustmatdb[j,4]=clustmatdb[j,4]+1
    }
    else if(classtag[which(allocatedclusterdb==j),][k]=='topic.trade')
    {
      clustmatdb[j,5]=clustmatdb[j,5]+1
    }
    else if(classtag[which(allocatedclusterdb==j),][k]=='topic.interest')
    {
      clustmatdb[j,6]=clustmatdb[j,6]+1
    }
    else if(classtag[which(allocatedclusterdb==j),][k]=='topic.ship')
    {
      clustmatdb[j,7]=clustmatdb[j,7]+1  
    }
    else if(classtag[which(allocatedclusterdb==j),][k]=='topic.money.fx')
    {
      clustmatdb[j,8]=clustmatdb[j,8]+1
    }
    else if(classtag[which(allocatedclusterdb==j),][k]=='topic.crude')
    {
      clustmatdb[j,2]=clustmatdb[j,2]+1
    }
    else if(classtag[which(allocatedclusterdb==j),][k]=='topic.wheat')
    {
      clustmatdb[j,9]=clustmatdb[j,9]+1
    }
  }
}
normclustdb=clustmatdb
for(l in 1:19)
{
  rsumclustdb=sum(clustmatdb[l,])
  normclustdb[l,]=clustmatdb[l,]/rsumclustdb
}
