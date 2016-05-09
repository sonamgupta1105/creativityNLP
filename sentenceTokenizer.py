from nltk.util import ngrams as getNgrams

def tokenize_andGetAvgJudgeScore(line):
	nltkFormatNgrams = []
	myFormatedNgrams = []
	totalJudgesScore = 0;
	
	valuesOnLine = line.split("\t")

	#The first 4 values in the valuesOnLine will be the judge scores
	for judgeID in range(4):
		totalJudgesScore += float(valuesOnLine[judgeID])
	# We calculate the average of the scores assigned by judges
	avgJudgesScore = totalJudgesScore / 4.0	

	sentenceWithKeywordAndGroup = valuesOnLine[4]
	tokensWithKeywordAndGroup = sentenceWithKeywordAndGroup.split()
	tokens = tokensWithKeywordAndGroup[2:]
	
	sentence = ""
	for token in tokens:
		sentence += str(token) + " "
	
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
	
	return (avgJudgesScore,myFormatedNgrams, sentence)
