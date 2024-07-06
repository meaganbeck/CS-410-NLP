import os
import numpy as np
import sys
import json
#this file calculates the probability matrix of the test files. 
#remove stop words??

#done with training data, store probabilities in output file? 

#alpha = 1

#vocab = []
##file = sys.argv[1]
#with open(file, 'r') as training_data: #tag data
#    for line in training_data:
#        if line.strip() != '':
#            line = line.split()
#            if line[0] not in vocab:
#                vocab.append(line[0])
#num_vocab = len(vocab)

def HMM(vocab, states, num_vocab, num_states):
    alpha = 0.1
    num_sequences = 0
    transition_dict = {}
    emission_dict = {}
    state_prevalence = {}
    initial_prob = {}

    for i in states:
        transition_dict[i] = {}
        state_prevalence[i] = 0
        initial_prob[i] = 0
        emission_dict[i] = {}
        
        for j in states:
            transition_dict[i][j] = 0
        for j in vocab:
            emission_dict[i][j] = 0
    
    
    training_file = sys.argv[1]

    with open(training_file, 'r') as training_data: #tag data
        prev_tag = None
        for line in training_data:
            print(line)
            line = line.strip()
            if line != '':
                line = line.split()
                word = line[0]
                tag = line[1]
                word_index = vocab.index(word)
                tag_index = states.index(tag)
                if prev_tag == '':# or states[prev_tag] == '.':
                    initial_prob[tag] +=1
                    num_sequences+=1

                    #print("i got here")
                else:
                    transition_dict[states[prev_tag]][states[tag_index]] +=1
                emission_dict[states[tag_index]][vocab[word_index]] +=1
                prev_tag = tag_index
                state_prevalence[tag]+=1

        
        #transition_file = open("transition.txt", "w")
        #emission_file = open("emission.txt", "w")


    with open("transition.json", 'w') as transition_file:
        dict_data = {}
        for prev_state in states:
            transition_prob = {}
            for curr_state in states:
                prob_smooth = ((alpha + transition_dict[prev_state][curr_state])/(num_states * alpha + state_prevalence[prev_state]))
                #prob_smooth = round(prob_smooth, 4)
                transition_prob[curr_state] = prob_smooth 
            dict_data[prev_state] = transition_prob
        json.dump(dict_data, transition_file )
    
    with open("emission.json", 'w') as emission_file:        
        dict_data = {}
        for tag in states:
            emission_prob = {}
            for word in vocab:
                prob_smooth = ((alpha + emission_dict[tag][word])/((num_vocab * alpha) + state_prevalence[tag]))
                #prob_smooth = round(prob_smooth, 4)
                emission_prob[word] = prob_smooth
            dict_data[tag] = emission_prob
        json.dump(dict_data, emission_file)
           

    with open("initial.json", 'w') as initial_file:
        initial = {}
        for tag in states:
            prob_smooth = initial_prob[tag]/num_sequences
            initial[tag] = prob_smooth
        json.dump(initial, initial_file)    
        #json_string = json.dumps(initial_prob[tag_index])
        #initial_file.write(json_string + '\n')

def main():
    if len(sys.argv) < 2:
        return
    states = []
    vocab = []
    training_file = sys.argv[1]

    with open(training_file, 'r') as training_data:
        for line in training_data:
            line = line.strip()
            #if line == '':
            #if line == '':
            if line != '':
             #   vocab.append('')
             #   states.append('')
            #else:
                word, tag = line.split()
                if word not in vocab:
                    vocab.append(word)
                if tag not in states:
                    states.append(tag)
    num_states = len(states)
    num_words = len(vocab)

    HMM(vocab, states, num_words, num_states)

if __name__ == "__main__":
    main()


