import numpy as np
from pickle import load
from keras.utils.vis_utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import LSTM, Bidirectional
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.merge import concatenate
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def load_dataset(filename):
    return load(open(filename, 'rb'))


def title_post_extractor(data_train):
    paragraph_list, post_list, title_list, des_list = [], [], [], []
    for title_post in data_train:
        paragraph_list.append(title_post[0])
        post_list.append(title_post[1])
        title_list.append(title_post[2])
        des_list.append(title_post[3])
    return [np.asarray(paragraph_list), np.asarray(post_list), np.asarray(title_list), np.asarray(des_list)]


def cnn_define_model(length, vocab_size):
    inputs1 = Input(shape=(length,))
    embedding1 = Embedding(vocab_size, 100)(inputs1)

    conv1 = Conv1D(filters=32, kernel_size=2, activation='relu')(embedding1)
    conv2 = Conv1D(filters=32, kernel_size=3, activation='relu')(embedding1)
    conv3 = Conv1D(filters=32, kernel_size=4, activation='relu')(embedding1)

    drop1 = Dropout(0.5)(conv1)
    drop2 = Dropout(0.5)(conv2)
    drop3 = Dropout(0.5)(conv3)

    pool1 = MaxPooling1D(pool_size=2)(drop1)
    pool2 = MaxPooling1D(pool_size=2)(drop2)
    pool3 = MaxPooling1D(pool_size=2)(drop3)

    conv11 = Conv1D(filters=64, kernel_size=2, activation='relu')(embedding1)
    conv12 = Conv1D(filters=64, kernel_size=3, activation='relu')(embedding1)
    conv13 = Conv1D(filters=64, kernel_size=4, activation='relu')(embedding1)

    drop11 = Dropout(0.5)(conv11)
    drop12 = Dropout(0.5)(conv12)
    drop13 = Dropout(0.5)(conv13)

    pool11 = MaxPooling1D(pool_size=4)(drop11)
    pool12 = MaxPooling1D(pool_size=4)(drop12)
    pool13 = MaxPooling1D(pool_size=4)(drop13)

    flat1 = Flatten()(pool11)
    flat2 = Flatten()(pool12)
    flat3 = Flatten()(pool13)

    merged1 = concatenate([flat1, flat2, flat3])
    dense1 = Dense(10, activation='relu')(merged1)
    output1 = Dense(1, activation='sigmoid')(dense1)

    model1 = Model(inputs=inputs1, outputs=output1)
    model1.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    # model1.summary()
    return model1

    # channel 1
    # inputs1 = Input(shape=(length,))
    # embedding1 = Embedding(vocab_size, 100, mask_zero=True)(inputs1)
    # conv1 = Conv1D(filters=32, kernel_size=4, activation='relu')(embedding1)
    # drop1 = Dropout(0.5)(conv1)
    # conv11 = Conv1D(filters=64, kernel_size=3, activation='relu')(drop1)
    # drop12 = Dropout(0.5)(conv11)
    # conv13 = Conv1D(filters=128, kernel_size=2, activation='relu')(drop12)
    # drop13 = Dropout(0.5)(conv13)
    # pool1 = MaxPooling1D(pool_size=2)(drop13)
    # flat1 = Flatten()(pool1)
    # # channel 2
    # inputs2 = Input(shape=(length,))
    # embedding2 = Embedding(vocab_size, 100)(inputs2)
    # conv2 = Conv1D(filters=32, kernel_size=4, activation='relu')(embedding2)
    # drop2 = Dropout(0.5)(conv2)
    # conv21 = Conv1D(filters=64, kernel_size=3, activation='relu')(drop2)
    # drop21 = Dropout(0.5)(conv21)
    # conv23 = Conv1D(filters=128, kernel_size=2, activation='relu')(drop21)
    # drop23 = Dropout(0.5)(conv23)
    # pool2 = MaxPooling1D(pool_size=2)(drop23)
    # flat2 = Flatten()(pool2)
    #
    # merged = concatenate([flat1, flat2])
    # dense1 = Dense(10, activation='relu')(merged)
    # outputs = Dense(1, activation='sigmoid')(dense1)
    # model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    # model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    # # print(model.summary())
    # plot_model(model, show_shapes=True, to_file='cnn_define_model.png')
    # return model

def cnn_seqential_model(length, vocab_size):
    # channel 1
    inputs1 = Input(shape=(length,))
    embedding1 = Embedding(vocab_size, 100, trainable=True)(inputs1)
    conv1 = Conv1D(filters=32, kernel_size=4, activation='relu')(embedding1)
    drop1 = Dropout(0.2)(conv1)
    conv11 = Conv1D(filters=64, kernel_size=3, activation='relu')(drop1)
    drop12 = Dropout(0.2)(conv11)
    pool1 = MaxPooling1D(pool_size=2)(drop12)
    lstm1 = Bidirectional(LSTM(units=128), merge_mode='concat')(pool1)


    # channel 2
    inputs2 = Input(shape=(length,))
    embedding2 = Embedding(vocab_size, 100)(inputs2)
    conv2 = Conv1D(filters=32, kernel_size=4, activation='relu')(embedding2)
    drop2 = Dropout(0.2)(conv2)
    conv21 = Conv1D(filters=64, kernel_size=3, activation='relu')(drop2)
    drop21 = Dropout(0.2)(conv21)
    pool2 = MaxPooling1D(pool_size=2)(drop21)
    lstm2 = Bidirectional(LSTM(units=128), merge_mode='concat')(pool2)

    merged = concatenate([lstm1, lstm2])

    dense1 = Dense(10, activation='relu')(merged)
    outputs = Dense(1, activation='sigmoid')(dense1)
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    plot_model(model, show_shapes=True, to_file='cnn_seq_define_model.png')
    return model

def seqential_model(length, vocab_size):
    # channel 1
    inputs1 = Input(shape=(length,))
    embedding1 = Embedding(vocab_size, 100)(inputs1)
    lstm1 = Bidirectional(LSTM(units=128), merge_mode='concat')(embedding1)
    # lstm11 = Bidirectional(LSTM(units=64, return_sequences=True), merge_mode='concat')(lstm1)
    # lstm12 = Bidirectional(LSTM(units=32), merge_mode='concat')(lstm11)


    # channel 2
    inputs2 = Input(shape=(length,))
    embedding2 = Embedding(vocab_size, 100)(inputs2)
    lstm2 = Bidirectional(LSTM(units=128), merge_mode='concat')(embedding2)
    # lstm21 = Bidirectional(LSTM(units=64, return_sequences=True), merge_mode='concat')(lstm2)
    # lstm22 = Bidirectional(LSTM(units=32), merge_mode='concat')(lstm21)

    inputs3 = Input(shape=(length,))
    embedding3 = Embedding(vocab_size, 100)(inputs3)
    lstm3 = Bidirectional(LSTM(units=128), merge_mode='concat')(embedding3)
    # lstm21 = Bidirectional(LSTM(units=64, return_sequences=True), merge_mode='concat')(lstm2)
    # lstm22 = Bidirectional(LSTM(units=32), merge_mode='concat')(lstm21)

    inputs4 = Input(shape=(length,))
    embedding4 = Embedding(vocab_size, 100)(inputs4)
    lstm4 = Bidirectional(LSTM(units=128), merge_mode='concat')(embedding4)
    # lstm21 = Bidirectional(LSTM(units=64, return_sequences=True), merge_mode='concat')(lstm2)
    # lstm22 = Bidirectional(LSTM(units=32), merge_mode='concat')(lstm21)

    merged = concatenate([lstm1, lstm2, lstm3, lstm4])

    dense1 = Dense(10, activation='relu')(merged)
    outputs = Dense(1, activation='sigmoid')(dense1)
    model = Model(inputs=[inputs1, inputs2, inputs3, inputs4], outputs=outputs)
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    # plot_model(model, show_shapes=True, to_file='cnn_seq_define_model.png')
    return model

# def ploting(history):
#     plt.plot(history.history['acc'])
#     plt.plot(history.history['val_acc'])
#     plt.title('model accuracy')
#     plt.ylabel('accuracy')
#     plt.xlabel('epoch')
#     plt.legend(['train', 'validation'], loc='upper left')
#     plt.show()
#     summarize history for loss
#     plt.plot(history.history['loss'])
#     plt.plot(history.history['val_loss'])
#     plt.title('model loss')
#     plt.ylabel('loss')
#     plt.xlabel('epoch')
#     plt.legend(['train', 'validation'], loc='upper left')
#     plt.show()


if __name__ == '__main__':
    title_post_padd, label, maximum_length, vocab_size = load_dataset('Clickbait_train_val_merge.pkl')
    maximum_length = 27
    train_target_post, test_target_post, train_label, test_label = train_test_split(title_post_padd, label, test_size=0.2, random_state=5)
    #
    # title, post = title_post_extractor(title_post_padd)

    train_paragraph, train_post, train_title, train_des = title_post_extractor(train_target_post)
    test_paragraph, test_post, test_title, test_des = title_post_extractor(test_target_post)

    cnn_model = cnn_define_model(length=maximum_length, vocab_size=vocab_size)
    cnn_history = cnn_model.fit(train_post, train_label, epochs=100, batch_size=32, validation_split=0.1, verbose=2)
    score = cnn_model.evaluate(test_post, test_label, batch_size=64)
    print("score:\t", score)
    # cnn_model = cnn_define_model(length=maximum_length, vocab_size=vocab_size)
    # # cnn_history = cnn_model.fit([title, post], label, epochs=10, batch_size=32, validation_split=0.2)
    # cnn_history = cnn_model.fit([train_title, train_post], train_label, epochs=5, batch_size=32, validation_data=([test_title, test_post],test_label))
    # score  = cnn_model.evaluate([test_title, test_post], test_label, batch_size=32)
    # print(score)

    # cnn_seq_model = cnn_seqential_model(length=maximum_length, vocab_size=vocab_size)
    # cnn_seq_history = cnn_seq_model.fit([train_title, train_post], train_label, epochs=10, batch_size=32, validation_data=([test_title, test_post],test_label))
    # seq_score = cnn_seq_model.evaluate([test_title, test_post], test_label, batch_size=32)
    # print(seq_score)
    # ploting(cnn_history)
    # # ploting(cnn_seq_history)
    # seq_model = seqential_model(length=maximum_length, vocab_size=vocab_size)
    # seq_history = seq_model.fit([train_paragraph, train_post, train_title, train_des], train_label, epochs=5, batch_size=10, validation_data=([test_paragraph, test_post, test_title, test_des],test_label))
    # rnn_seq_score = seq_model.evaluate([test_paragraph, test_post, test_title, test_des], test_label, batch_size=10)
    # print(rnn_seq_score)