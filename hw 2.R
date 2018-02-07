# install packages:
install.packages('RPostgreSQL')

# load packages:
library(RPostgreSQL)
library(arules)

# loads the PostgreSQL driver
drv <- dbDriver("PostgreSQL")

# login information
un <- 'zma3282' # your database user name
pw <- 'zma3282_pw' # your database password
sn <- 'zma3282_schema' # your schema name 

# creates a connection to the iems308 database
# "con" will be used in each connection to the database
con <- dbConnect(drv, dbname = "iems308",
                 host = "gallery.iems.northwestern.edu", 
                 port = 5432,
                 user = un, password = pw)

#subset data #
#select top 5 stores with highest revenue and purchase records only limit to 100000#
trnsact = dbGetQuery(con, "select c1,c2,c3,c4,c6 from pos.trnsact 
                     where pos.trnsact.c7='P' and 
                     pos.trnsact.c2 in ('504','9304','7507','1607','8402')limit 100000")

# rename column #
colnames(trnsact)<- c('sku','store','register','trannum','saledate')

# data preparation #
trnsact$sku<- as.factor(trnsact$sku)
trnsact$store<- as.numeric(trnsact$store)
trnsact$register<- as.numeric(trnsact$register)
trnsact$trannum<- as.numeric(trnsact$trannum)
trnsact$saledate<- as.Date(trnsact$saledate) 
trnsact$busket<- paste(trnsact$store,trnsact$register,trnsact$trannum,trnsact$saledate,collapse=NULL,sep=",")
trnsact$busket<- as.factor(trnsact$busket)
trans<- data.frame(trnsact$busket,trnsact$sku)

#read transaction#
transaction<-write.csv(trans,file="pos.trnsact.csv",row.names=FALSE)
transaction<-read.transactions("pos.trnsact.csv",cols=c(1,2),format="single",rm.duplicates=TRUE)

#find rules#
rules<-apriori(transaction,parameter=list(supp=0.00009,conf=0.2,minlen=2))
summary(rules)

rules1<-apriori(transaction,parameter=list(supp=0.00009,conf=0.5,minlen=2))

#optimize rules#
subset.matrix<- is.subset(rules1,rules1,sparse = FALSE)
subset.matrix[lower.tri(subset.matrix,diag=T)]<-NA
redundant<- colSums(subset.matrix,na.rm = T)>=1
which(redundant)

rules.p<- rules1[!redundant]
rules.p<- sort(rules.p,by='lift')
inspect(rules.p)

#plot rules#
plot(rules.p)

