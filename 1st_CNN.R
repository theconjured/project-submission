library(mxnet)


train.x <- train[,-1]
train.y <- train[,1]
train.x <- t(train.x/255)
test.x <- t(test/255)

m2.data <- mx.symbol.Variable("data")

m2.conv1 <- mx.symbol.Convolution(m2.data, kernel=c(5,5), num_filter=16)
m2.bn1 <- mx.symbol.BatchNorm(m2.conv1)
m2.act1 <- mx.symbol.Activation(m2.bn1, act_type="relu")
m2.pool1 <- mx.symbol.Pooling(m2.act1, pool_type="max", kernel=c(2,2), stride=c(2,2))
m2.drop1 <- mx.symbol.Dropout(m2.pool1, p=.5)

m2.conv2 <- mx.symbol.Convolution(m2.drop1, kernel=c(3,3), num_filter=32)
m2.bn2 <- mx.symbol.BatchNorm(m2.conv2)
m2.act2 <- mx.symbol.Activation(m2.bn2, act_type="relu")
m2.pool2 <- mx.symbol.Pooling(m2.act2, pool_type="max", kernel=c(2,2), stride=c(2,2))
m2.drop2 <- mx.symbol.Dropout(m2.pool2, p=0.5)
m2.flatten <- mx.symbol.Flatten(m2.drop2)

m2.fc1 <- mx.symbol.FullyConnected(m2.flatten, num_hidden=1024)
m2.act3 <- mx.symbol.Activation(m2.fc1, act_type="relu")

m2.fc2 <- mx.symbol.FullyConnected(m2.act3, num_hidden=512)
m2.act4 <- mx.symbol.Activation(m2.fc2, act_type="relu")

m2.fc3 <- mx.symbol.FullyConnected(m2.act4, num_hidden=256)
m2.act5 <- mx.symbol.Activation(m2.fc3, act_type="relu")

m2.fc4 <- mx.symbol.FullyConnected(m2.act5, num_hidden=10)
m2.softmax <- mx.symbol.SoftmaxOutput(m2.fc4)

train.array <- train.x
dim(train.array) <- c(28,28,1,ncol(train.x))
test.array <- test.x
dim(test.array) <- c(28,28,1,ncol(test.x))

m2 <- mx.model.FeedForward.create(m2.softmax, 
                                  X = train.array, 
                                  y = train.y,
                                  num.round = 200, # This many will take a couple of hours on a CPU
                                  array.batch.size = 500,
                                  array.layout="colmajor",
                                  learning.rate = 0.01,
                                  momentum = 0.91,
                                  wd = 0.00001,
                                  eval.metric = mx.metric.accuracy,
                                  initializer = mx.init.uniform(0.07),
                                  epoch.end.callback = mx.callback.log.train.metric(1, log)
)

m2.preds <- predict(m2, test.array)
m2.preds.value <- max.col(t(m2.preds))-1

submission <- data.frame(ImageId=1:ncol(test.x), Label=m2.preds.value)
write.csv(submission, file="submission_kaggle_digits_CNN", row.names=FALSE, quote=FALSE)

head(submission)
