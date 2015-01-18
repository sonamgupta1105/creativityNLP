import sys
import os
import sqlite3
from nltk.util import ngrams as getNgrams
from sklearn.ensemble import RandomForestRegressor 
from sklearn.externals import joblib

import feature1 as f1
import feature2 as f2
import feature3 as f3
import feature4and5 as f4and5


def main(argv):

	#load random forest model
	trainedRandomForest = joblib.load('trainedRandomForestModel/creativeWritingRF_trainedModel.pkl')

	databaseName = "googleNGram_BACKUP_afterCrashOfUnigramsAnd68BigramFiles.db"

	#open connection to database
	if(os.path.isfile(databaseName)):
		connDB = sqlite3.connect(databaseName)
		#This is used to be able to handle other language symbols. forces the database to use unicode strings
		connDB.text_factory = str
	else:
		print "DATABASE FILE: " + databaseName + " does not exist!"
		return

	while(True):
		sentence = str(raw_input('Please enter a sentence (to quit type "!QUIT!"): '))
		if sentence == "!QUIT!":
			print "Thank you! Goodbye..."
			break
		
		tempListOfFeatures = []	
			
		#get the ngrams for the current sentence
		sentenceNgrams = tokenize(sentence)

		numWordsInSentence = len(sentenceNgrams[0])

		tempListOfFeatures.extend( f1.getValues(sentenceNgrams, numWordsInSentence, connDB) )
 		tempListOfFeatures.append( f2.getValue(sentenceNgrams[0], numWordsInSentence, connDB) )
		#We return tagged sentences because we need POS tags for 4th feature as well 
		(feature3value,tagged_sentence) = f3.getValueAndTaggedSentence(sentence)
		tempListOfFeatures.append( feature3value ) 
		tempListOfFeatures.extend( f4and5.getValues(tagged_sentence) )

		#with the tempListOfFeatures we now the random forest to get prediction
		predictedValue = trainedRandomForest.predict(tempListOfFeatures)

		if predictedValue < 3:
			print "That was not very creative. Your score: " + str(predictedValue[0])
		elif predictedValue < 6:
			print "That was okay, but I am sure you can do better. Your score: " + str(predictedValue[0])
		else:
			print "Impressive! Your score: " + str(predictedValue[0])

	return

def tokenize(sentence):
	nltkFormatNgrams = []
	myFormatedNgrams = []

	tokens = sentence.split()
	
	for n in range(3):
		nltkFormatNgrams.append(getNgrams(tokens,n+1))
	
	for j in range(len(nltkFormatNgrams)):
		#creating a separate list for the unigrams, bigrams, and trigrams by looping here
		myFormatedNgrams.append([])
		a = nltkFormatNgrams[j]
		allGramsOfTheCurrentType = list(a)

		for i in range( len(allGramsOfTheCurrentType)):
    
			x = list(allGramsOfTheCurrentType[i])
		
			#building up the current ngram by appending all the words which were previously listed in a list
			#into single strings with the words separated by spaces
			#Before: ex1. ["example", "of", "trigram"]      ex2. ["a", "bigram"]
			#After:  ex1. "example of trigram"              ex2. "a bigram"
			tempStr = ""
			for y in range(len(x)-1):
				tempStr += str(x[y]) + " "
			tempStr += str(x[len(x)-1])
			myFormatedNgrams[j].append(tempStr)		
	
	return myFormatedNgrams




if __name__ == "__main__":
	# The reason for this is that the main function above and all of its contents only
	#      will be run when this python script is executed with the command of
	#      python myScriptName.py
    	#      Rather than without this main setup check the program would be run even if this 
    	#      file were to be imported by another script via the import command.
	main(sys.argv)
