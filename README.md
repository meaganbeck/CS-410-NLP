This program uses data from "Gender Ambiguous Pronoun" Kaggle competition with dat from https://github.com/google-research-datasets/gap-coreference.
I included the data sets I used, as I altered the provided data. 


#Installation
Libraries needed are nltk, pickle, and numpy
The program automatically downloads names, words, punkt, averaged_perceptron_tagger, and maxent_ne_chunker from nltk

Unfortunately, the data-names are hardcoded, and will, in its current state, only work for the specific files as labeled. 

To run training:
	python3 corefResolution.py gap-training-data.tsv classifier.pkl

To run testing:
	python3 corefResolution.py gap-test-data.tsv classifier.pkl


Testing the data should output pair candidates with the highest probability per paragraph in the data set into "output.txt". 
This would be compaired to the gold standard data with "evaluation.py", that was never completed, but would have ideally outputted
the F1 score 
