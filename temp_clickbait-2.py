import json
import os, re
import pickle
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import preprocessor as p
p.set_options(p.OPT.URL, p.OPT.EMOJI, p.OPT.MENTION, p.OPT.HASHTAG, p.OPT.SMILEY, p.OPT.RESERVED)



def train_validation_gen(training_instance_file_path, training_truth_file_path):
    attributes = ['id', 'postTimestamp', 'postText', 'postMedia', 'targetTitle', 'targetDescription', 'targetKeywords', 'targetParagraphs', 'targetCaptions', 'truthMedian', 'truthClass']

    def postText_cleaning(postText):
        tweet = p.clean(postText)
        tweet = re.sub(r"https?", ' ', tweet)
        tweet = re.sub(r"([^A-Za-z0-9\s\.\,:'’])", ' ', tweet)
        tweet = re.sub(r" +", ' ', tweet)
        tweet = re.sub(r"’", "'", tweet)
        # tweet = re.sub(r"\.", " .", tweet)
        # tweet = re.sub(r",", " ,", tweet)
        # tweet = re.sub(r":", " :", tweet)
        tweet = re.sub(r" +", ' ', tweet)
        return tweet


    def targetParagraphs_cleaning(targetParagraphs):
        new_str = str()
        for tp in targetParagraphs:
            clean_tweet = p.clean(tp)
            tweet = re.sub(r"https?", ' ', clean_tweet)
            tweet = re.sub(r"([^A-Za-z0-9\s\.\,:'’])", ' ', tweet)
            tweet = re.sub(r" +", ' ', tweet)
            tweet = re.sub(r"’", "'", tweet)
            # tweet = re.sub(r"\.", " .", tweet)
            # tweet = re.sub(r",", " ,", tweet)
            # tweet = re.sub(r":", " :", tweet)
            tweet = re.sub(r" +", ' ', tweet)
            new_str = new_str + " " + tweet
            new_str = new_str.strip()
        return new_str


    def targetTitle_cleaning(targetTitle):
        tweet = p.clean(targetTitle)
        tweet = re.sub(r"https?", ' ', tweet)
        tweet = re.sub(r"([^A-Za-z0-9\s\.\,:'’])", ' ', tweet)
        tweet = re.sub(r" +", ' ', tweet)
        tweet = re.sub(r"’", "'", tweet)
        # tweet = re.sub(r"\.", " .", tweet)
        # tweet = re.sub(r",", " ,", tweet)
        # tweet = re.sub(r":", " :", tweet)
        tweet = re.sub(r" +", ' ', tweet)
        return tweet

    def targetDescription_cleaning(targetDescription):
        tweet = p.clean(targetDescription)
        tweet = re.sub(r"https?", ' ', tweet)
        tweet = re.sub(r"([^A-Za-z0-9\s\.\,:'’])", ' ', tweet)
        tweet = re.sub(r" +", ' ', tweet)
        tweet = re.sub(r"’", "'", tweet)
        # tweet = re.sub(r"\.", " .", tweet)
        # tweet = re.sub(r",", " ,", tweet)
        # tweet = re.sub(r":", " :", tweet)
        tweet = re.sub(r" +", ' ', tweet)
        return tweet

# final data creates a list "Paragraphs_post_title_des" which hold
#     [[targetParagraph, postText, targetTitle, targetDescription], [targetParagraph, postText, targetTitle, targetDescription],...]
    def final_data(truth_id_list, truth_class_list, id_list, postText_list, targetTitle_list, targetDescription_list, targetParagraphs_list):
        Paragraphs_post_title_des, class_list = [], []
        for i in range(len(truth_id_list)):
            temp_list = []
            for j in range(len(id_list)):
                if truth_id_list[i] == id_list[j]:
                    try:
                        temp_list.append(targetParagraphs_list[j].strip())
                    except:
                        temp_list.append(targetParagraphs_list[j])
                    temp_list.append(postText_list[j].strip())
                    temp_list.append(targetTitle_list[j].strip())
                    temp_list.append(targetDescription_list[j].strip())
                    class_list.append(truth_class_list[j])
            Paragraphs_post_title_des.append(temp_list)
            del(temp_list)
        return [Paragraphs_post_title_des, class_list]


    instance_file = open(training_instance_file_path,'r')
    id_list, postText_list, targetParagraphs_list, targetTitle_list, targetDescription_list = [], [], [], [], []

    for line in instance_file:
        line = json.loads(line)
        id = line[attributes[0]]
        postText = line[attributes[2]][0]
        targetTitle = line[attributes[4]]
        targetDescription = line[attributes[5]]
        targetParagraphs = line[attributes[7]]

        if len(postText) == 0:
            postText = '0'
        if len(targetTitle) == 0:
            targetTitle = '0'
        if len(targetDescription) == 0:
            targetDescription = '0'
        if len(targetParagraphs) == 0:
            targetParagraphs = ['0']

        postText = postText_cleaning(postText)
        targetParagraphs = targetParagraphs_cleaning(targetParagraphs)
        targetTitle = targetTitle_cleaning(targetTitle)
        targetDescription = targetDescription_cleaning(targetDescription)

        id_list.append(id)
        postText_list.append(postText)
        targetParagraphs_list.append(targetParagraphs)
        targetTitle_list.append(targetTitle)
        targetDescription_list.append(targetDescription)


    truth_file = open(training_truth_file_path, 'r')
    truth_id_list, truth_class_list = [], []
    for line in truth_file:
        line = json.loads(line)
        truth_id = line[attributes[0]]
        truth_clickbait = line[attributes[10]]

        if truth_clickbait == 'clickbait':
            truth_clickbait = 1
        else:
            truth_clickbait = 0
        truth_id_list.append(truth_id)
        truth_class_list.append(truth_clickbait)

    Paragraphs_post_title_des, class_list = final_data(truth_id_list, truth_class_list, id_list, postText_list, targetTitle_list, targetDescription_list, targetParagraphs_list)
    return [Paragraphs_post_title_des, class_list]

def merge_train_val(train_Paragraphs_post_title_des, train_class_list, val_Paragraphs_post_title_des, val_class_list):
    i = 0
    for _ in train_Paragraphs_post_title_des:
        val_Paragraphs_post_title_des.append(train_Paragraphs_post_title_des[i])
        val_class_list.append(train_class_list[i])
        i = i + 1
    return [val_Paragraphs_post_title_des, val_class_list]



def encoding_set(data_train, label_train):

    def create_tokenizer(train_lines_list):
        lines = [sent.strip() for line in train_lines_list for sent in line]
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(lines)
        return tokenizer

    def max_length(train_lines_list):
        count = 0
        tp_len, pt_len, t, d = [], [], [], []
        for line in train_lines_list:
            targetParagraph = len(line[0].split())
            if targetParagraph == 37731:
                print(line[0].split())
            postText = len(line[1].split())
            title = len(line[2].split())
            des = len(line[3].split())
            count = count + 1
            tp_len.append(targetParagraph)
            pt_len.append(postText)
            t.append(title)
            d.append(des)
        # print('Total sentence\t:\t', count)
        # print('Maximum target Paragraph Length:\t',sorted(tp_len)[-5:])
        # print('Maximum POst text Length:\t',sorted(pt_len)[-5:])
        # print('Maximum Title Length:\t',sorted(t)[-5:])
        # print('Maximum Description Length:\t',sorted(d)[-5:])
        return max(max(tp_len), max(pt_len), max(t), max(d))

    def encode_text(tokenizer, train_lines, length):
        train_target_post_padded_list = []
        for target_post1 in train_lines:
            target_encode1 = tokenizer.texts_to_sequences([str(target_post1[0])])
            post_encode1 = tokenizer.texts_to_sequences([str(target_post1[1])])
            title_encode1 = tokenizer.texts_to_sequences([str(target_post1[2])])
            des_encode1 = tokenizer.texts_to_sequences([str(target_post1[3])])

            target_padded1 = pad_sequences(target_encode1, maxlen=length, padding='post', truncating='post')
            post_padded1 = pad_sequences(post_encode1, maxlen=length, padding='post')
            title_padded1 = pad_sequences(title_encode1, maxlen=length, padding='post')
            des_padded1 = pad_sequences(des_encode1, maxlen=length, padding='post')

            train_target_post_padded_list.append([target_padded1[0], post_padded1[0], title_padded1[0], des_padded1[0]])
        return train_target_post_padded_list


    tokenizer = create_tokenizer(data_train)
    maximum_length = max_length(data_train)
    # print(tokenizer.word_docs)

    vocab_size = len(tokenizer.word_index) + 1
    print('Maximum length is: %d' % maximum_length)
    print('Vocabulary size: %d' % vocab_size)

    maximum_length = 657
    Paragraphs_post_title_des_padd = encode_text(tokenizer, data_train, maximum_length)

    with open('temp_Clickbait_train_val_merge.pkl', 'wb') as db:
        pickle.dump([Paragraphs_post_title_des_padd, label_train, maximum_length, vocab_size],db)


if __name__ == '__main__':

    folders = [folder for folder in os.listdir(os.getcwd()) if os.path.isdir(folder) and not folder.startswith('.')]
    train_folder_path = os.getcwd() + '/' + folders[0]
    val_folder_path = os.getcwd() + '/' + folders[1]
    training_instance_file_path = os.path.join(os.getcwd() + '/' + folders[0], 'instances.jsonl')
    training_truth_file_path = os.path.join(os.getcwd() + '/' + folders[0], 'truth.jsonl')

    val_instance_file_path = os.path.join(os.getcwd() + '/' + folders[1], 'instances.jsonl')
    val_truth_file_path = os.path.join(os.getcwd() + '/' + folders[1], 'truth.jsonl')

    train_Paragraphs_post_title_des, train_class = train_validation_gen(training_instance_file_path, training_truth_file_path)
    val_Paragraphs_post_title_des, val_class = train_validation_gen(val_instance_file_path, val_truth_file_path)
    Paragraphs_post_title_des_list, class_list = merge_train_val(train_Paragraphs_post_title_des, train_class, val_Paragraphs_post_title_des, val_class)
    encoding_set(Paragraphs_post_title_des_list, class_list)
