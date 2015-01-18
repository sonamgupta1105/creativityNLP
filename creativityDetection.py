###########################################
# Sonam Gupta                                     
# ssg154@psu.edu                             
# COMP594: Project
# Fall 2014                    
# How Creatively Do You Write?                                                         
#
###########################################
#This is where we make all the imports
import sys
import os
import sqlite3

#My code files to import
import setupAndCommands_DB as database
import feature1 as f1
import feature2 as f2
import feature3 as f3
import feature4and5 as f4and5
import sentenceTokenizer as sentTokenizer
import crossValidation
import generatePlot as gp
import reviewDataset

#########################################

#Defined constants

#if firstRunOfProgram is 1, this indicates that the database needs to be created
#else it is 0 which means that database is already existing
firstRunOfProgram = 0

databaseName = "googleNGram_uni10bi10_v1.db"

googleNGramDirectoryPath = "/home/sonam/Desktop/recentStuff/" #"~/Desktop/TESTING_GOOGLE_NGRAM_SCRIPT/" 
dataSetPath = "WisconsinCreativeWriting_editedForReadingIntoProgram.txt"
outputAvgJudgesScoreFile = "outputFiles/avgJudgesScore_perSentence.txt"
outputFeaturesFile = "outputFiles/features_perSentence.txt"

############################################################################################### 
#This class will be to extract the features from each of the sentences in the given dataset

############################################################################################### 
class featureExtractor:

	# Following is the constructor of the class. It gets called when creating an instance of the class
	def __init__(self, databaseConnection, datasetPath):
        	# self denotes local variables to the class
		self.connDB = databaseConnection

		self.datasetForExtractingFeatures = datasetPath

        	#Here we will store in a list of feature values for each of the sentences
        	#Each sentence will have a list of k values and we will have this 
		#list containing all of those lists (for the k feature values). 
		self.featuresPerSentence = []

		self.avgJudgesScorePerSentence = []

		self.linesOfFile = []		

    #END init constructor
############################################
	def run(self):	
	
		datasetFile = open(str(self.datasetForExtractingFeatures), "r")
	
		for line in datasetFile.readlines():
			
			self.linesOfFile.append(line)

			#reset the temp list of features for the current sentence
			tempListOfFeatures = []	
			
			#get the ngrams for the current sentence
			(avgJudgeScore,sentenceNgrams, sentence) = sentTokenizer.tokenize_andGetAvgJudgeScore(line)
			#sentenceNgrams format by example "This is my example sentence." : 
			#[['This', 'is', 'my', 'example', 'sentence.'], ['This is', 'is my', 'my example', 'example sentence.'],    
			#...(continued to next line)
			#  ['This is my', 'is my example', 'my example sentence.']]			

			#######################
			##Testing
			##print sentenceNgrams
			##print avgJudgeScore
			##print "\n\n\n"
			#######################			

			#Since the unigrams are stored in the first list of sentenceNgrams this will return the
			#number of unigrams, which is the number of words in the sentence
			numWordsInSentence = len(sentenceNgrams[0])
			
			#Get the feature values for the current sentence
			# We extend for getting the values of f1 because feature1 returns three values
			tempListOfFeatures.extend( f1.getValues(sentenceNgrams, numWordsInSentence, self.connDB) )
 			tempListOfFeatures.append( f2.getValue(sentenceNgrams[0], numWordsInSentence, self.connDB) )
			#We return tagged sentences because we need POS tags for 4th feature as well 
			(feature3value,tagged_sentence) = f3.getValueAndTaggedSentence(sentence)
			tempListOfFeatures.append( feature3value ) 
			tempListOfFeatures.extend( f4and5.getValues(tagged_sentence) )
			
			#append the current sentence's list of features into the list of all sentences
			self.featuresPerSentence.append(tempListOfFeatures)

			self.avgJudgesScorePerSentence.append(avgJudgeScore)
		#END sentence of datasetFile
	
		return (self.avgJudgesScorePerSentence, self.featuresPerSentence, self.linesOfFile)



############################################################################################### 
############################################################################################### 
############################################################################################### 

def main(argv):
	# This is where we will run the main program

	# This will be the driver of all the other functions calling them and creating instances 
	# of things such as the featureExtractor above.

	if firstRunOfProgram:
		#We check if the database already exists...if so, delete it and create a new one
#REMOVE THIS COMMENT		if(os.path.isfile(databaseName)):
			#then we delete it and create a new file
#REMOVE THIS COMMENT			os.remove(databaseName)
		#Connect to our database
		#since the database hasn't been created (or it was just deleted) it will be created now	
		connDB = sqlite3.connect(databaseName)
		#This is used to be able to handle other language symbols. forces the database to use unicode strings
		connDB.text_factory = str

		#We create the tables and place the google ngram data in the tables
#REMOVE THIS COMMENT		database.setupTablesInDB(connDB)
#REMOVE THIS COMMENT		database.storeGoogleNGramFiles(googleNGramDirectoryPath, connDB)
		
		#Commit all the changes we made to the database
#REMOVE THIS COMMENT		connDB.commit()
	else:#This is not the first run of the program and we already have the database setup
		# Connect to our database
		connDB = sqlite3.connect(databaseName)
		#This is used to be able to handle other language symbols. forces the database to use unicode strings
		connDB.text_factory = str

	# We create an instance of the feature extractor class
	myFeatExtract = featureExtractor(connDB,dataSetPath)
	
	# We run our feature extractor on our dataset and store the output into 
	# a list of lists, where the nested lists are of the following form [feat1Value, ... ,featKValue] for our k features
	# and the outer list is just a list of our nested lists. Each of the lines (ie. sentences) in our dataset
	# will be a list of features (ie inner/nested lists) and the entire dataset as a whole is a list of the nested 
	# lists which each of the nested lists represented a sentence
	# example for 2 sentences with 4 features: 
	# [ [s1.f1, s1.f2, s1.f3, s1.f4] , [s2.f1, s2.f2, s2.f3, s2.f4] ]
	# we also get the avg score between the four judges for each of the sentences and have them stored in a list
	# example for the two sentences above the avg score might be:
	# [ 2.4, 8.2 ]
	(listOfAvgJudgesScore,listOfFeatureLists,linesOfFile) = myFeatExtract.run()
	
	#Quick save the avg judge scores and the lists of features to a temp output file
	#Each line will be the avg judge score for a sentence
	outputFile_avgJudgesScore = open(outputAvgJudgesScoreFile, 'w')
	#Each line will be a tab separated list of the features for that sentence
	outputFile_features = open(outputFeaturesFile, 'w')

	for avgScore in listOfAvgJudgesScore:
		outputFile_avgJudgesScore.write(str(avgScore) + "\n")

	outputFile_features.write("f1.1 \t f1.2 \t f1.3 \t f2 \t ner \t verb \t noun \t adj \t distinct \n")
	for featureList in listOfFeatureLists:
		for i in range(len(featureList)-1):
			outputFile_features.write(str(format(featureList[i], '.3f')) + "\t")
		outputFile_features.write(str(format(featureList[len(featureList)-1], '.3f')) + "\n")
	
	outputFile_avgJudgesScore.close()
	outputFile_features.close()

	#Run k fold cross validation on the dataset
	(foldsAvgJudgesScore, foldsFeatureLists, foldsLinesOfFile) = crossValidation.generateFolds(listOfAvgJudgesScore, listOfFeatureLists, linesOfFile)	
	overallPredictedScores = crossValidation.run(foldsAvgJudgesScore, foldsFeatureLists, foldsLinesOfFile)


	#create a random forest trained on the entire wisconsin dataset to be used for review dataset predictions
	fullDatasetRandomForest = crossValidation.generateAndSaveRF_model(listOfAvgJudgesScore,listOfFeatureLists)

	#predict the values for the review dataset using the trained random forest
	reviewDataset.predict(fullDatasetRandomForest,connDB)

	#Create a plot of the results compared to average judge scores
	gp.createPlot(listOfAvgJudgesScore,overallPredictedScores)	
	
	connDB.close()	

if __name__ == "__main__":
	#      The reason for this is that the main function above and all of its contents only
	#      will be run when this python script is executed with the command of
	#      python myScriptName.py
    	#      Rather than without this main setup check the program would be run even if this 
    	#      file were to be imported by another script via the import command.
	main(sys.argv)


