import tflearn
from tflearn.data_utils import to_categorical, pad_sequences
from tflearn.datasets import imdb


# prepare imdb dataset, download it, and load it
train, valid, test = imdb.load_data('imdb.pkl', 10000, valid_portion=0.1)

print 'load data success!'
print 'train: \n', len(train[0])
print 'valid: \n', len(valid[0])
print 'test: \n', len(test[0])

# split dataset
train_X, train_y = train
valid_X, valid_y = valid
test_X, test_y = test

# print 'train[1]', train_X[1]
# print 'train[2]', train_X[2]
#
# print 'test[0]', test_y[0]
# print 'test[1]', test_y[1]
# print 'test[2]', test_y[2]
# print 'test[3]', test_y[3]

# sequences padding
train_X = pad_sequences(train_X, maxlen=100, value=0.)
valid_X = pad_sequences(valid_X, maxlen=100, value=0.)
test_X = pad_sequences(test_X, maxlen=100, value=0.)
# to categorical
train_y = to_categorical(train_y, nb_classes=2)
valid_y = to_categorical(valid_y, nb_classes=2)
test_y = to_categorical(test_y, nb_classes=2)


# print 'train[1]', train_X[1]
# print 'train[2]', train_X[2]
# print 'test[0]', test_y[0]
# print 'test[1]', test_y[1]
# print 'test[2]', test_y[2]
# print 'test[3]', test_y[3]

# define neural networks
net = tflearn.input_data(shape=[None, 100])
net = tflearn.embedding(net, input_dim=10000, output_dim=128)
net = tflearn.lstm(net, 128, dropout=0.8)
net = tflearn.fully_connected(net, 2, activation='softmax')
net = tflearn.regression(net, optimizer='adam', loss='categorical_crossentropy', learning_rate=0.01)

# train the model
model = tflearn.DNN(net, tensorboard_verbose=3, tensorboard_dir='./tmp/tf_log/')
model.fit(train_X, train_y, n_epoch=10, validation_set=(valid_X, valid_y), batch_size=32, show_metric=True)

print 'fit success!'
model.save('test1.dnn.model')
print 'save success!'

print 'predict!'
predict_result = model.predict(test_X[:100])
print 'predict result:\n\n', predict_result
print 'real result:\n\n', test_y[:100]