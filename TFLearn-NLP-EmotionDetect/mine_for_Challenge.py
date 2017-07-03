import pandas as pd
import numpy as np
import tflearn
from tflearn.data_utils import pad_sequences, to_categorical, VocabularyProcessor


MAX_TITLE_SIZE = 30


# parse data frame dataset
data_frame = pd.read_csv('ign.csv')
Y_score_phrase = data_frame[['score_phrase']]
X_title = data_frame[['title']]

# print X_title[0:100]
# print data_frame['title'][:100]
print X_title.size  # 18625 total lines


# clean and pre-process of data
vocabulary_processor = VocabularyProcessor(max_document_length=MAX_TITLE_SIZE)
i_generator = vocabulary_processor.fit_transform(raw_documents=data_frame['title'])
X_All_word_ids = np.array(list(i_generator))
for i in X_All_word_ids[:100]:
    print i
vocabulary_processor.save('vocabulary_processor.dict')

# replace origin data frame to after word id mapping iterable object
X_title = X_All_word_ids


# split the data
n_split_train = int(X_title.size*0.8)
n_split_validate = int(n_split_train*0.8)
print "Split will be at index 0 - %d - %d - %d" % (n_split_validate, n_split_train, X_title.size)
train_X = X_title[:n_split_validate]
validate_X = X_title[n_split_validate:n_split_train]
test_X = X_title[n_split_train:]

train_Y = Y_score_phrase[:n_split_validate]
validate_Y = Y_score_phrase[n_split_validate:n_split_train]
test_Y = Y_score_phrase[n_split_train:]


# print train_X[0:100]
# train_X = pad_sequences(train_X, maxlen=10, value=0.)


# define network
net = tflearn.input_data(shape=[None, MAX_TITLE_SIZE])
net = tflearn.embedding(net, input_dim=10000, output_dim=128)
net = tflearn.lstm(net, n_units=256, dropout=0.8)
net = tflearn.fully_connected(net, n_units=8, activation='softmax')
net = tflearn.regression(net, optimizer='adam', loss='categorical_crossentropy', learning_rate=0.01)


# training
model = tflearn.DNN(network=net, tensorboard_verbose=3, tensorboard_dir="./tmp/tf_log/")
model.fit(X_inputs=train_X, Y_targets=train_Y, n_epoch=10,
          validation_set=(validate_X, validate_Y), show_metric=True, batch_size=32)

