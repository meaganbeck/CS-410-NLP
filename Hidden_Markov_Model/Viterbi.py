import numpy as np
import os
import sys
import json

def load_json_file(file_path):
    try:
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)
        return data
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in file '{file_path}'.")
        return None

def Viterbi(transition_dict, emission_dict, vocab, num_words, initial_prob, test_data):
    states = list(transition_dict.keys())

    
    T = num_words
    N = len(states)
    viterbi = {}
    for i in range(N):
        viterbi[i] = {}
        for j in range(T): #FIXME maybe range of test_data
            viterbi[i][j] = 0
    
    backpointer = np.zeros((N,T), dtype=int)
    
    for s in range(N): 
        state = states[s]
        viterbi[s][0] = initial_prob[state] * emission_dict[state][test_data[0]]

        backpointer[s][0] = 0
        
    for t in range(1, T):
    
        for s in range(N):
            max_prob = 0
            max_state = 0
            #state= states[s] 
            for prev_s in range(N):
                if test_data[t] not in emission_dict[states[s]]:
                    prob = 0.1
                else:
                    prob = viterbi[prev_s][t-1] * transition_dict[states[prev_s]][states[s]] * emission_dict[states[s]][test_data[t]]
                    #print(transition_dict[states[prev_s]][state])
                if prob > max_prob:
                    max_prob = prob
                    max_state = prev_s
            viterbi[s][t] = max_prob
            backpointer[s][t] = max_state
    
    bestpathprob = 0
    bestpathpointer = 0

    for s in range(N):
        if viterbi[s][T-1] > bestpathprob:
            bestpathprob = viterbi[s][T-1]
            bestpathpointer = s
            
    bestpath = [bestpathpointer]
    #bestpathnew = [bestpathpointer]
    for t in range(T-1, 0, -1):
        bestpathpointer = backpointer[bestpathpointer][t]
        if test_data[t] == '':
            bestpath.insert(1, '')
        else:
            bestpath.insert(0, states[bestpathpointer])
    
    return bestpath, bestpathprob



def main():
    if len(sys.argv) < 2:
        return
    states = []
    vocab = [] #unique words 
    test_data =  [] #total words
    test_file = sys.argv[1]
    initial_prob = []

    with open(test_file, 'r') as data:
        for line in data:
            line = line.strip()
            if line != '':
                if line not in vocab:
                    vocab.append(line)
            test_data.append(line)
    vocab_len = len(vocab)
    num_words = len(test_data)

    transition_dict = {}
    emission_dict = {}
    
    transition_dict = load_json_file("transition.json")
    if transition_dict is None:
        return

    emission_dict = load_json_file("emission.json")
    if emission_dict is None:
        return
    
    initial_prob = load_json_file("initial.json")
    if initial_prob is None:
        return
    bestpath, bestpathprob = Viterbi(transition_dict, emission_dict, vocab, num_words, initial_prob, test_data)
    
    with open("wsj_23.pos", "w") as output:
        for el in range(num_words-1):
            output.write(str(test_data[el]) +  '\t' + str(bestpath[el]) + '\n')
        output.write('\n')
       # output.write(" ".join(str(bestpath)) + "\n")

if __name__ == "__main__":
    main()








