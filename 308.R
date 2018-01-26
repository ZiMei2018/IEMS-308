#define work directory
setwd("C:\\Users\\admin\\Desktop")

#library
library(cluster)
library(factoextra)
library(data.table)

#read data
data<- fread("Medicare_Provider_Util_Payment_PUF_CY2015.txt",header = TRUE,sep="\t",fill = TRUE,na.strings = c("","NA"))

#prepare data
data1<- na.omit(data)
sample<- data1[sample(1:nrow(data1),9575,replace= FALSE),]
scaled.data<- scale(sample[,c(23:26)])

#explore data
qqnorm(sample$average_Medicare_allowed_amt,ylab = 'Average allowed amount')
qqnorm(sample$average_Medicare_payment_amt,ylab = 'Average payment amount')
qqnorm(sample$average_submitted_chrg_amt,ylab = 'Average submitted amount')
qqnorm(sample$average_Medicare_standard_amt, ylab = 'Average standard amount')
plot(sample$average_Medicare_allowed_amt)
plot(sample$average_Medicare_payment_amt)
plot(sample$average_submitted_chrg_amt)
plot(sample$average_Medicare_standard_amt)

# find clusters
fviz_nbclust(scaled.data,kmeans,method = 'wss')

#k-means cluster
k.cluster<- kmeans((sample[,c(23:26)]),centers = 4)

#plot with submitted amount and allowed amount
plot(sample$average_Medicare_allowed_amt~sample$average_submitted_chrg_amt,col = k.cluster$cluster)

#plot with payrate and submitted amount
payrate<- sample$average_Medicare_allowed_amt/sample$average_submitted_chrg_amt
data.test<- data.frame(sample,payrate)
cluster1<- kmeans((data.test[,c(23:27)]),centers = 4)
plot(payrate~average_submitted_chrg_amt,sample,col=cluster1$cluster)


