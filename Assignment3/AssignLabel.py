import nltk
import pickle
from nltk.classify import MaxentClassifier
import numpy
import sys
from nltk.corpus import stopwords

nltk.download('stopwords')


def read_file(filename):
    new_data = []
    with open(filename, 'r') as new_doc:
        for line in new_doc:
            line = line.strip()
            new_line = []
            if line != "":
                #FixMe and what i'll need here later
                segments = line.split()
                word = segments[0] if len(segments) > 0 else None
                pos = segments[1] if len(segments) > 1 else None
                chunk = segments[2] if len(segments) > 2 else None
                #^FIXME
                #new_line.append((word, pos, chunk))
                new_data.append((word,pos,chunk))
            elif line == "":
                new_data.append((""))
    return new_data
    #it's formatted as a list containing tuples of data

def extract_features(line, position, prev_tag):
    features = {} #create binary features. is uppercase or not, is digit or not
    
    stop_words = set(stopwords.words('english'))
    
    if len(line) >= 3:
        curr_word = line[0]
        pos = line[1]
        chunk = line[2]
        
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

def format_extraction(new_data):
    
    position = 0
    prev_tag = None
    features_list = []# or []
    line_num = 0

    for line in new_data: #each "line" is a tuple
        if len(line) > 2:
            chunk = line[2] if len(line) > 2 else None
            pos = line[1] if len(line) > 1 else None
            word = line[0] if len(line) > 0 else None
            data_dict = extract_features(line, position, prev_tag)
            features_list.append(data_dict) 
        else:
            features_list.append("")
        position+=1
        #prev_tag = curr_tag
        line_num +=1

    return features_list
    #.pos-chunk-name 


#def label(read_classifier, new_data):
#    result = read_classifier.classify(new_data)

def main():
    input_data = sys.argv[1]
    new_data = read_file(input_data)
    

    extracted_data = format_extraction(new_data)
    classifier = sys.argv[2]
 
    with open(classifier,'rb') as f:
        read_classifier = pickle.load(f)
    
    with open('test.name', 'w') as output_file:
        n = 0
        for token in extracted_data:
            if token == "":
                output_file.write('\n')
            else:
                result = read_classifier.classify(token)
                output_file.write(new_data[n][0] + ' ' + result + '\n')
            n+=1


    #output classifier data
if __name__=="__main__":
    main()

