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
print Y_score_phrase['score_phrase']


# clean and pre-process of data, only pre-process X data, using VocabularyProcessor
vocabulary_processor = VocabularyProcessor(max_document_length=MAX_TITLE_SIZE)
i_generator = vocabulary_processor.fit_transform(raw_documents=data_frame['title'])
X_All_word_ids = np.array(list(i_generator))
for i in X_All_word_ids[:100]:
    print i
vocabulary_processor.save('vocabulary_processor.dict')
print 'dictionary had saved!'

# replace origin data frame to after word id mapping iterable object
X_title = X_All_word_ids
print 'X_title:'
print X_title


print 'Y_score_phrase: size:'+str(len(Y_score_phrase))
# pre-process Y data
labels = {}
labels_statistics = {}
Y_score_phrase_nums = []
for y_item in Y_score_phrase['score_phrase']:
    print y_item
    if y_item not in labels:
        labels[y_item] = len(labels)
        labels_statistics[y_item] = 0
    Y_score_phrase_nums.append(labels[y_item])
    labels_statistics[y_item] += 1

print 'labels: ', labels
print 'label statistics: ', labels_statistics


# split the data
n_split_train = int(len(X_title)*0.8)
n_split_validate = int(n_split_train*0.8)
print "Split will be at index 0 - %d - %d - %d" % (n_split_validate, n_split_train, len(X_title))
train_X = X_title[:n_split_validate]
validate_X = X_title[n_split_validate:n_split_train]
test_X = X_title[n_split_train:]

train_Y = Y_score_phrase[:n_split_validate]
validate_Y = Y_score_phrase[n_split_validate:n_split_train]
test_Y = Y_score_phrase[n_split_train:]


# print train_X[0:100]
# train_X = pad_sequences(train_X, maxlen=10, value=0.)
# change the data
# train_Y = to_categorical(train_Y, nb_classes=8)
# print train_Y


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

