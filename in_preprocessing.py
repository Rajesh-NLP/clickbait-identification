import json
import os, re
import pickle
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences





def train_validation_gen(training_instance_file_path, training_truth_file_path):
    attributes = ['id', 'postTimestamp', 'postText', 'postMedia', 'targetTitle', 'targetDescription', 'targetKeywords', 'targetParagraphs', 'targetCaptions', 'truthMedian', 'truthClass']

    def postText_cleaning(postText):
        text = postText.split(' ')
        new_text = str()
        for t in text:
            if t.startswith('http'):
                pass
            else:
                new_text = new_text+' '+ t
        new_text.strip()
        # print(new_text)
        return new_text


    def targetParagraphs_cleaning(targetParagraphs):
        for paragraph_element in targetParagraphs:
            paragraph_element = re.sub(r"[\[\]]",'', paragraph_element)
            new_str = str()
            for word in paragraph_element.split():
                if word.startswith('http') or word.startswith('https'):
                    pass
                else:
                    new_str = new_str +' '+ word
            new_str.lstrip(' ')
            return new_str


    def targetTitle_cleaning(targetTitle):
        return targetTitle

    def targetDescription_cleaning(targetDescription):
        return targetDescription

    def final_data(truth_id_list, truth_class_list, id_list, postText_list, targetTitle_list, targetDescription_list, targetParagraphs_list):
        Paragraphs_post_title_des, class_list = [], []
        for i in range(len(truth_id_list)):
            temp_list, temp1_list = [], []
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
            del(temp1_list)
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

        # print(id,'\t',targetParagraphs)

        postText = postText_cleaning(postText)
        targetParagraphs = targetParagraphs_cleaning(targetParagraphs)
        targetTitle = targetTitle_cleaning(targetTitle)
        targetDescription = targetDescription_cleaning(targetDescription)
        # print(targetTitle)
        # print(postText)

        id_list.append(id)
        postText_list.append(postText)
        targetParagraphs_list.append(targetParagraphs)
        targetTitle_list.append(targetTitle)
        targetDescription_list.append(targetDescription)
    # print(id_list)



    truth_file = open(training_truth_file_path, 'r')
    truth_id_list, truth_class_list = [], []
    for line in truth_file:
        line = json.loads(line)
        truth_id = line[attributes[0]]
        truth_clickbait = line[attributes[10]]
        # print(truth_clickbait)
        if truth_clickbait == 'clickbait':
            truth_clickbait = 1
        else:
            truth_clickbait = 0
        truth_id_list.append(truth_id)
        truth_class_list.append(truth_clickbait)
        # print(truth_clickbait)
    # print(truth_id_list)



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
        lines = list()
        for line in train_lines_list:
            for sent in line:
                try:
                    lines.append(sent.strip())
                except:
                    pass
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(lines)
        return tokenizer

    def max_length(train_lines_list):
        tp_len, pt_len, t, d = [], [], [], []
        for line in train_lines_list:
            try:
                targetParagraph = len(line[0].split())
            except:
                targetParagraph = 0
            try:
                postText = len(line[1].split())
            except:
                postText = 0
            try:
                title = len(line[2].split())
            except:
                title = 0
            try:
                des = len(line[3].split())
            except:
                des = 0

            tp_len.append(targetParagraph)
            pt_len.append(postText)
            t.append(title)
            d.append(des)
        print("Maximum target Paragraph length:", max(tp_len))
        print("Maximum post Text length:", max(pt_len))
        print("Maximum Title length:", max(t))
        print("Maximum Description length:", max(d))
        return max(max(tp_len), max(pt_len), max(t), max(d))



    def encode_text(tokenizer, train_lines, length):
        length = 27
        train_target_post_padded_list = []
        for target_post1 in train_lines:
            target_encode1 = tokenizer.texts_to_sequences([str(target_post1[0])])
            post_encode1 = tokenizer.texts_to_sequences([str(target_post1[1])])
            title_encode1 = tokenizer.texts_to_sequences([str(target_post1[2])])
            des_encode1 = tokenizer.texts_to_sequences([str(target_post1[3])])

            target_padded1 = pad_sequences(target_encode1, maxlen=length, padding='post')
            post_padded1 = pad_sequences(post_encode1, maxlen=length, padding='post')
            title_padded1 = pad_sequences(title_encode1, maxlen=length, padding='post')
            des_padded1 = pad_sequences(des_encode1, maxlen=length, padding='post')

            train_target_post_padded_list.append([target_padded1[0], post_padded1[0], title_padded1[0], des_padded1[0]])
        return train_target_post_padded_list


    tokenizer = create_tokenizer(data_train)
    maximum_length = max_length(data_train)

    vocab_size = len(tokenizer.word_index) + 1
    print('Maximum length is: %d' % maximum_length)
    print('Vocabulary size: %d' % vocab_size)

    Paragraphs_post_title_des_padd = encode_text(tokenizer, data_train, maximum_length)

    with open('Clickbait_train_val_merge.pkl', 'wb') as db:
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
