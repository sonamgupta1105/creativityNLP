from sklearn.ensemble import RandomForestRegressor 
from sklearn.metrics import mean_square_error
from sklearn.externals import joblib
from math import sqrt
import numpy as np

def run(foldsAvgJudgesScore, foldsFeatureLists, foldsLinesOfFile):

	numTreesInForest = 50
	outputFilePartialName = "outputFiles/predictedCreativeScores_onFoldNum_"

	#These will hold the values from all the sentences
	#We add the first folds predicted scores when k=1
	#Similarly we add the second folds predicted scores when k=2
	overallAvg_rSquaredScores = []
	overallAvg_rootMeanSquareErrorScores = []
	overallPredictedScores = []

	#create a model for all 5 folds
	for k in range(5):
		#We let the kth index to be the testing index
		tempTrainFolds_AvgJudgesScore = []
		tempTrainFolds_FeatureLists = []
		tempTrainFolds_LinesOfFile = []
		for j in range(5):
			if (j == k):
				tempTestFold_AvgJudgesScore = foldsAvgJudgesScore[j]	
				tempTestFold_FeatureLists = foldsFeatureLists[j]
				tempTestFold_LinesOfFile = foldsLinesOfFile[j]
			else:
				tempTrainFolds_AvgJudgesScore.extend(foldsAvgJudgesScore[j])
				tempTrainFolds_FeatureLists.extend(foldsFeatureLists[j])
				tempTrainFolds_LinesOfFile.extend(foldsLinesOfFile[j])

		#now that we have the test and training sets we create our random forest
		
		#Creating a random forest of decision trees and having the number of processes running set to 
		#-1 where it will run X processes generating X trees at a time where X is the number of cores on the cpu
		tempForest = RandomForestRegressor(n_estimators=numTreesInForest, n_jobs=-1)		
		
		#Training
		tempForest.fit(tempTrainFolds_FeatureLists,tempTrainFolds_AvgJudgesScore)

		#Testing
		predictedCreativeScores = tempForest.predict(tempTestFold_FeatureLists)
		#predictedCreativeScores are the values from the kth fold (ie our current test fold)
		#and we add them to our overall predicted, which after doing this for all k folds we will have
		#predicted all the values
		overallPredictedScores.extend(predictedCreativeScores)		

		rSquaredScore = tempForest.score(tempTestFold_FeatureLists,tempTestFold_AvgJudgesScore)
		rootMeanSquareErrorScore = sqrt(mean_square_error(tempTestFold_AvgJudgesScore, predictedCreativeScores))
		#we add the rsquared and rootmeansquareerror scores just like we did for the predicted scores so that
		#after we have tested all k folds we will have the scores for each fold and therefore each sentence
		overallAvg_rSquaredScores.append(rSquaredScore)
		overallAvg_rootMeanSquareErrorScores.append(rootMeanSquareErrorScore)

		#Calculate the feature importances
		#The below was slightly modified from an example on the scikit learns webpage
		#http://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html#example-ensemble-plot-forest-importances-py
		featureImportances = tempForest.feature_importances_
		if featureImportances != None:
			indices = np.argsort(featureImportances)[::-1]

			tempFeatures = []
			for feature in range(len(tempTestFold_FeatureLists[0])):
				tempFeatures.append( [indices[feature] , featureImportances[indices[feature]]] )
		

		#Write the predictedCreativeScores to an output file
		tempOutputFileName = outputFilePartialName + str(k) + ".txt"
		tempOutput = open(tempOutputFileName, 'w')
	
		#First lines of the file are the scores and then the format of the file		
		tempOutput.write("r squared (r^2) score: " + str(rSquaredScore) + "\n")
		tempOutput.write("root mean square error (RMSE) score: " + str(rootMeanSquareErrorScore) + "\n")
		
		if featureImportances != None:		
			#Feature importances printed
			for feat in range(len(tempFeatures)-1):
				tempOutput.write(str(tempFeatures[feat][0]) + "|" + str(tempFeatures[feat][1]) + ", ")
			#printing the last feature with new line
			tempOutput.write(str(tempFeatures[len(tempFeatures)][0]) + "|" + str(tempFeatures[len(tempFeatures)][1]) + "\n")			
			
		tempOutput.write("\nFormat for the below: \nPredicted Score (tab) AvgJudge Score (tab) line from their dataset file \n\n")	

		for i in range(len(predictedCreativeScores)):
			tempOutput.write(str(predictedCreativeScores[i]) + "\t" + str(tempTestFold_AvgJudgesScore[i]) + "\t" + str(tempTestFold_LinesOfFile[i]) + "\n")

		#Done with this fold so we close the file
		tempOutput.close()
	return overallPredictedScores






def generateAndSaveRF_model(listOfAvgJudgesScore,listOfFeatureLists):

	#We create a random forest training on the entire Wisconsin dataset
	#This will be used for our review dataset predictions as well as the
	#online version of the program
	fullDatasetRandomForest = RandomForestRegressor(n_estimators=50, n_jobs=-1)		
		
	#Training on all the data
	fullDatasetRandomForest.fit(listOfFeatureLists,listOfAvgJudgesScore)

	joblib.dump(fullDatasetRandomForest, "trainedRandomForestModel/creativeWritingRF_trainedModel.pkl")

	return fullDatasetRandomForest

def generateFolds(listOfAvgJudgesScore, listOfFeatureLists, linesOfFile):

	#We will try a 5 fold cross validation
	foldsAvgJudgesScore = []	
	foldsFeatureLists = []
	foldsLinesOfFile = []

	groupSize = len(listOfAvgJudgesScore) / 5

	groupStart = 0
	groupEnd = groupSize
		

	foldsAvgJudgesScore.append( listOfAvgJudgesScore[groupStart:groupEnd])
	foldsFeatureLists.append( listOfFeatureLists[groupStart:groupEnd])
	foldsLinesOfFile.append( linesOfFile[groupStart:groupEnd])	

	groupStart = groupEnd
	groupEnd += groupSize

	foldsAvgJudgesScore.append( listOfAvgJudgesScore[groupStart:groupEnd])
	foldsFeatureLists.append( listOfFeatureLists[groupStart:groupEnd])
	foldsLinesOfFile.append( linesOfFile[groupStart:groupEnd])	

	groupStart = groupEnd
	groupEnd += groupSize

	foldsAvgJudgesScore.append( listOfAvgJudgesScore[groupStart:groupEnd])
	foldsFeatureLists.append( listOfFeatureLists[groupStart:groupEnd])
	foldsLinesOfFile.append( linesOfFile[groupStart:groupEnd])	

	groupStart = groupEnd
	groupEnd += groupSize

	foldsAvgJudgesScore.append( listOfAvgJudgesScore[groupStart:groupEnd])
	foldsFeatureLists.append( listOfFeatureLists[groupStart:groupEnd])
	foldsLinesOfFile.append( linesOfFile[groupStart:groupEnd])	

	groupStart = groupEnd

	foldsAvgJudgesScore.append( listOfAvgJudgesScore[groupStart:])
	foldsFeatureLists.append( listOfFeatureLists[groupStart:])
	foldsLinesOfFile.append( linesOfFile[groupStart:])	

	return (foldsAvgJudgesScore, foldsFeatureLists, foldsLinesOfFile)
