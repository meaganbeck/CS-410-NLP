import sys
import nltk
import numpy
import pickle
import os
import re
from nltk.classify import MaxentClassifier
from nltk.corpus import stopwords
from nltk.corpus import names
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk import ne_chunk
from nltk.tree import Tree
from collections import defaultdict
nltk.download('names')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

def read_file(filename):
    """Reads in file from inputted data and formats information """
    new_data = []
    with open(filename, 'r') as new_doc:
        for line in new_doc:
            line = line.strip()
            new_line = []
            if line != "":
                segments = line.split('\t')
                if len(segments) == 11:
                    ID, text, pronoun, pronoun_offset, A, A_offset, A_coref, B, B_offset, B_coref = segments[:10]
                    new_data.append((ID, text, pronoun, pronoun_offset, A, A_offset, A_coref, B, B_offset, B_coref))
                elif len(segments) == 1:
                    text = segments[0]
                    new_data.append((text))
            elif line == "":
                new_data.append((""))
                
    return new_data
    #it's formatted as a list containing tuples of data

def pos(new_data):
    """Determines part-of-speech tags for each element in the dataset"""
    tag_data = []
    tagged = ()

    for line in new_data[1:]:
        if len(line) > 1 and len(line[1]) > 1:
            text = line[1]
            tokenize = nltk.word_tokenize(text)
            tagged = nltk.pos_tag(tokenize) #list of tuples of word, tag        
            
            tag_data.append(tagged)
    return tag_data 

def get_chunks(new_data):
    """Chunks the data for each line in the dataset"""
    chunked_data = []
    
    # Skipping the first line (header)
    for line_index, line in enumerate(new_data[1:], start=1):  # Enumerate from 1 to align with line numbering
        if len(line) > 1 and len(line[1]) > 1:
            text = line[1]
            tokens = word_tokenize(text)
            pos_tags = pos_tag(tokens)
            chunked = ne_chunk(pos_tags)
            cont_chunk = []

            curr_chunk = []
            pos_tags = []
            start_position = None
            total_length = 0

            for index, i in enumerate(chunked):
                if isinstance(i, Tree):  
                    if not curr_chunk:
                        start_position = total_length  
                    curr_chunk.append(" ".join([token for token, pos in i.leaves()]))

                total_length += len(tokens[index]) + 1 

                if not isinstance(i, Tree) and curr_chunk:
                    named_entity = " ".join(curr_chunk)
                    if named_entity.strip():  
                        cont_chunk.append({
                            "entity": named_entity,
                            "line_position": line_index,
                            "character_position": start_position
                        })
                    curr_chunk = []
                    start_position = None

            if curr_chunk:
                named_entity = " ".join(curr_chunk)
                if named_entity.strip():  # Check if the named entity is non-empty and valid
                    cont_chunk.append({
                        "entity": named_entity,
                        "line_position": line_index,
                        "character_position": start_position
                    })

            if cont_chunk:
                chunked_data.append(cont_chunk)

    return chunked_data


def extract_features(new_data, tagged, chunked):
    """Call for feature extractions after gathering all appropriate information"""
    position = 0
    
    #best_candidates = []
    all_candidates = {}
    
    #For each line in the dataset, complete this loop
    #for line_index in range(1, 2): #each "line" is a tuple
    for line_index in range(1, len(new_data)): #each "line" is a tuple
        curr_tagged = tagged[line_index-1]
        antecedents = chunked[line_index-1]
        line = new_data[line_index]
        all_candidates[line_index] = [] 
        if len(line) == 10:
            label, text, pronoun, pronoun_offset, A, A_offset, A_coref, B, B_offset, B_coref = line[:10]
            if A_coref == "TRUE":
                gold_pair = ((A, A_offset),(pronoun,pronoun_offset))
            elif B_coref == "TRUE":
                gold_pair = ((B, B_offset),(pronoun,pronoun_offset))
            elif A_coref == "FALSE" and B_coref == "FALSE":
                gold_pair = 0
            else:
                gold_pair = None
        #Get list of anaphora present
            anaphora_list = []
            for index in range(len(curr_tagged)):
                if curr_tagged[index][1] in ('PRP', 'PRP$'):
                    anaphora_list.append(curr_tagged[index])
       
       #Get feature set                 
            pair_features_list = pair_features(text, antecedents, anaphora_list,curr_tagged) #pair features, plus            
        

        #Determine proper candidate pairs
            for pair in pair_features_list:
                candidate_pair = candidate_pairs(text, pair, gold_pair) 
                if candidate_pair == "valid":
                    #if line_index in all_candidates:
                    all_candidates[line_index].append((pair, candidate_pair))
                    #else:
                        #all_candidates[line_index] = [(pair, candidate_pair)]
                #all_candidates.update({line_index: (pair, candidate_pair)})
               # all_candidates[line_index] = append(pair, candidate_pair))
                    
    return all_candidates

def pair_features(line, antecedent_list, anaphora_list, tagged):
    """Creates the feature set for each pair of antecedent and anaphora"""
    #antecedent list and anaphora list okay
    sentences = line.split('.')
    pair_features_list = []
    anaphora_index = {}
    antecedent_index = {}

    for ant in antecedent_list:
        ant_prev = 0
        for a in anaphora_list:
            a_prev = 0
            pair_features = {}

        #Sentence Distance
            sentence_count = 0
            for sent in sentences:
                if ant["entity"] in sent:
                    if a[0] in sent:
                        pair_features["sentence_distance"] = 0
                    else:
                        sentence_count+=1
                if a[0] in sent:
                    pair_features["sentence_distance"] = sentence_count

        #Antecedent and Anaphora
            pair_features["antecedent"] = ant["entity"]
            pair_features["anaphora"] = a[0]

        #Index
            #pair_features["anaphora_index"] 
            pattern = re.compile(r'\b' + re.escape(a[0]) + r'\b')
            match = pattern.search(line, a_prev)
            if match != None:
                pair_features["anaphora_index"] = match.start()
            #line.index(r'\b' + a[0] + r'\b', a_prev)
            else: 
                pair_features["anaphora_index"] = 0

            pattern = re.compile(r'(?<!w)' + re.escape(ant["entity"]) + r'(?!\w)')
            match = pattern.search(line, ant_prev)
            if match != None:
                pair_features["antecedent_index"] = match.start()
            else: 
                pair_features["antecedent_index"] = 0
            #pair_features["antecedent_index"] = line.index(ant["entity"], ant_prev)
            a_prev = pair_features["anaphora_index"]
            ant_prev = pair_features["antecedent_index"]

        #Mention Distance
            pair_features["mention_distance"] = int(pair_features["anaphora_index"]) - int(pair_features["antecedent_index"])

        #Number 
            if a[0].lower() in ("them", "themselves", "we", "ourselves", "they", "their", "us"):
                pair_features["number"] = "plural"
            else:
                pair_features["number"] = "singular"
        #Gender
            full_name = ant["entity"].split()
            if (full_name[0] in names.words("male.txt")) and (a[0].lower() in ("he", "him", "his", "himself")):
                pair_features["gender"] = "male"
            elif (full_name[0] in names.words("female.txt")) and (a[0].lower() in ("she", "her", "hers", "herself")):
                pair_features["gender"] = "female"
            elif (full_name[0] in names.words("male.txt")) and (a[0].lower() in ("she", "her", "hers", "herself")):
                pair_features["gender"] = "invalid"
            elif (full_name[0] in names.words("female.txt")) and (a[0].lower() in ("he", "him", "his", "himself")):
                pair_features["gender"] = "invalid"

            else:
                pair_features["gender"] = "unknown"

            pair_features_list.append(pair_features)
    return pair_features_list

def candidate_pairs(line, pair, gold_pair):
    """Determines the validity of each antecedent-anaphora pair"""
    
    if gold_pair == 0:
        if (pair["gender"] in ("male", "female")) and (pair["number"] in ("plural", "singular"))and (pair["mention_distance"] > 0):
            candidate_pair = "valid"
        else: 
            candidate_pair = "invalid"

    elif gold_pair == None:
        candidate_pair = "invalid"

    elif (pair["gender"] in ("male", "female")) and (pair["number"] in ("plural", "singular"))and (pair["mention_distance"] > 0):
        if ((pair["anaphora"] == gold_pair[1][0]) and (pair["anaphora_index"] == gold_pair[1][1]) and (pair["antecedent"] == gold_pair[0][0]) and (str(pair["antecedent_index"]) == gold_pair[0][1])):
            candidate_pair = "valid"
        else:
            candidate_pair = "invalid"
    else:
        candidate_pair = "invalid"
    
    return candidate_pair 


def mention_ranking(all_candidates, classifier, output_file):
    """Calculates the probability of each candidate pair and ranks them accordingly"""
    anaphora_candidates = defaultdict(list)

    assignments = {}
    #all_candidates[0] = [()()()]
    #all_candidates[1] = [()()()]
    for line in all_candidates.items():
        for pair in line:
            print(pair)
        #for features, label in line:
            prob_dist = classifier.prob_classify(pair[0])
            score = prob_dist.prob("valid")
    
            anaphora = pair[0]["anaphora"]
            antecedent = pair[0]["antecedent"]

            anaphora_candidates[anaphora].append((antecedent,score))
        #rank_candidates.append((features,score))
        

    #Assign best anaphora-antecedent pairs
        for anaphora, candidates in anaphora_candidates.items():
        # Sort candidates based on score
            candidates.sort(key=lambda x: x[1], reverse=True)

            best_antecedent, best_score = candidates[0]
            assignments[anaphora] = best_antecedent
            
            output_file.write(anaphora + '\t' + best_antecedent + '\t' + str(best_score)+ '\n')
            output_file.flush()
    return assignments

def main():
    """ADD STUFF HERE"""
    input_data = sys.argv[1]
    #error checking here
    new_data = read_file(input_data)
    tagged_data = pos(new_data)
    chunked_data = get_chunks(new_data)
    #tagged is a list of lists
    
    all_candidates = extract_features(new_data, tagged_data, chunked_data)

    classifier_file = sys.argv[2]
    
    #Error checking here
    file_path = '/' + str(classifier_file)
    if str(input_data) == "gap-training-data.tsv":
        #if os.path.exists(file_path) == False:
        if "train" in str(classifier_file):
            #ranked_candidates = mention_ranking(all_candidates, classifier)
            classifier = MaxentClassifier.train(all_candidates, algorithm='iis', max_iter=10)

            with open(classifier_file, 'wb') as f:
                pickle.dump(classifier, f)

    elif str(input_data) == "gap-test-data.tsv":
        with open(classifier_file,'rb') as f:
            read_classifier = pickle.load(f)
    
        with open('output.txt', 'w') as output_file:
            assignment = mention_ranking(all_candidates, read_classifier,output_file) 
            
            #output_file.write(assignment)


if __name__=="__main__":
    main()
