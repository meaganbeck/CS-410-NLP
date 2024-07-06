import sys
import nltk 
import numpy
import os
from nltk.classify import MaxentClassifier
import pickle
from nltk.corpus import stopwords

nltk.download('stopwords')

def read_file(filename):
    training_data = []
    
    with open(filename, 'r') as training_doc:
        for line in training_doc:
            line = line.strip()
            training_line = []
            if line != "":
                segments = line.split()
                word = segments[0] if len(segments) > 0 else None
                pos = segments[1] if len(segments) > 1 else None
                chunk = segments[2] if len(segments) > 2 else None
                ner_label = segments[3] if len(segments) > 3 else None
                training_line.append((word, pos, chunk, ner_label))
                training_data.append((word,pos,chunk,ner_label))
    return training_data
    #it's formatted as a list containing tuples of data


def extract_features(line, position, prev_tag):
    features = {} #create binary features. is uppercase or not, is digit or not
    stop_words = set(stopwords.words('english'))


    if len(line) > 3:
        curr_word = line[0]
        pos = line[1]
        chunk = line[2]
        curr_tag = line[3]
        features["first_letter"] = curr_word[0].isupper()
        if curr_word in stop_words:
            features["stop_word"] = True
        else:
            features["stop_word"] = False
        if pos == "NNP":
            features["POS_NNP"] = True
        else:
            features["POS_NNP"] = False
        if pos == "JJ":
            features["POS_JJ"] = True
        else:
            features["POS_JJ"] = False
        if chunk == "I-NP":
            features["I-NP"] = True
        else:
            features["I-NP"] = False
        if prev_tag == "I-PER" or prev_tag == "O-PER":
        #OR
            features["prev_tag_name"] = True
        else:
            features["prev_tag_name"] = False
    return features #returns a dictionary
#features = {"first_letter": 1, "prev_tag_name": 0} in this example

def format_extraction(output_file, training_data):
    
    position = 0
    prev_tag = None
    features_list = []# or []
    line_num = 0

    for line in training_data: #each "line" is a tuple
        curr_tag = line[3] if len(line) > 3 else None #the NER
        chunk = line[2] if len(line) > 2 else None
        pos = line[1] if len(line) > 1 else None
        word = line[0] if len(line) > 0 else None

        data_tuple = ((extract_features(line, position, prev_tag), curr_tag))
        features_list.append(data_tuple) 
        
        position+=1
        prev_tag = curr_tag
        line_num +=1

    return features_list
    #.pos-chunk-name 

def main():
    
    filename = sys.argv[1] #train.pos-chunk-name
    training_data = read_file(filename)
        #list of tuples (word, pos, chunk, ner)
    output_file = sys.argv[2] #output_features.txt
    #format_output calls extract features
    features = format_extraction(output_file, training_data)
    classifier = MaxentClassifier.train(features, algorithm='iis', max_iter=10)
    
    #classifier.pkl
    with open(output_file,'wb') as f:
        pickle.dump(classifier, f)
    
if __name__=="__main__":
    main()
