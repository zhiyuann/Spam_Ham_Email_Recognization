from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import random
import os
import nltk
import numpy
import math

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn


def load_data(dir):
	list = []
	for file in os.listdir(dir):
		with open(dir+'/'+file,'rb') as f:
			body = f.read().decode('utf-8', errors='ignore').splitlines()

			list.append(' '.join(body))
			
	return list

def preprocess(text):
	text=text.lower()
	tokenizer = RegexpTokenizer(r'[a-z]+')
	tokens = tokenizer.tokenize(text)
# then return a list of tokens (words)
	lemmatizer=WordNetLemmatizer()
	stoplist=stopwords.words('english')
	tokens=[lemmatizer.lemmatize(t) for t in tokens]
	tokens=[t for t in tokens if not t in stoplist]
	return tokens

SpamDict={}
HamDict={}

def featurizeTokens(tokens,is_spam):
	#for(tokens,label) in train:
	if(is_spam):

		for temp in tokens:
			if temp in SpamDict:
				SpamDict[temp]=SpamDict[temp]+1
			else:
				SpamDict.update({temp:1})
	else:
		for temp in tokens:
			if temp in HamDict:
				HamDict[temp]=HamDict[temp]+1
			else:
				HamDict.update({temp:1})
	
		#featurizeTokens(tokens,label=='spam')

BASE_DATA_DIR='/home/student/Downloads/EE364/enron1' # fill me in
# load and tag data

ham = [(text, 'ham') for text in load_data(BASE_DATA_DIR + '/ham')]
spam = [(text, 'spam') for text in load_data(BASE_DATA_DIR +'/spam')]

#The above part cites from the EE364 Computer Problem 2 Instruction
all = ham + spam
print(len(ham))
print(len(spam))

all = [(preprocess(text),label) for (text,label) in all]
random.shuffle(all)

splitp = 0.8
train=all[:int(splitp*len(all))]
test=all[int(splitp*len(all)):]

SpamDict={}
HamDict={}

for(tokens,label) in train:

	featurizeTokens(tokens,label=='spam')
	
	

total_spam_word_count = sum(SpamDict.values())
total_ham_word_count = sum(HamDict.values())

spam_in_train_count = sum([label=='spam' for (tokens, label) in train])
ham_in_train_count = sum([label=='ham' for (tokens, label) in train])

train_count = len(train)

P_spam = spam_in_train_count / train_count
P_ham = ham_in_train_count/train_count

#print(HamDict)
SpamPred = []
HamPred = []

pred_label = numpy.array([])
true_label = numpy.array([])

for (tokens,label) in test:
	#print(tokens)
	
	P_word_spam = numpy.log(P_spam)
	P_word_ham = numpy.log(P_ham)
	
	for word in tokens:

		if  word in SpamDict:
			P_word_spam +=  numpy.log(SpamDict[word] / len(spam))
		else:
			P_word_spam += numpy.log(1.0 / (total_spam_word_count + len(SpamDict) +1))

		if word in HamDict:
			P_word_ham += numpy.log(HamDict[word]/len(ham))
		else:
			P_word_ham += numpy.log(1/(total_ham_word_count + len(HamDict)+1))

	true_label = numpy.append(true_label, label)
	if(P_word_spam)>(P_word_ham):
		pred_label = numpy.append(pred_label, "spam")
	else:
		pred_label = numpy.append(pred_label, "ham")

num_spam_test = sum([label=='spam' for (tokens, label) in test])
num_ham_test = sum([label=='ham' for (tokens, label) in test])

# Compute the metrics.		
TruePosRate = numpy.sum(np.logical_and(true_label == "spam", pred_label == "spam"))/ (num_spam_test) 
print(TruePosRate)
FalseNegRate = numpy.sum(np.logical_and(true_label == "spam", pred_label == "ham"))/ (num_spam_test) 
print(FalseNegRate)
TrueNegRate = numpy.sum(np.logical_and(true_label == "ham", pred_label == "ham"))/ (num_ham_test) 
FalsePosRate = numpy.sum(np.logical_and(true_label == "ham", pred_label == "spam"))/ (num_ham_test) 


df = pd.DataFrame([[TruePosRate, FalsePosRate], [FalseNegRate,
TrueNegRate]])
fig = plt.figure()
ax = sn.heatmap(100*df, vmin=0, vmax=100, cmap='Blues',annot=True, fmt='.2f', annot_kws={"size":16}, linewidths=0.5)
ax.set_xlabel('Truth')
ax.set_ylabel('Prediction')
ax.set_xticklabels(['spam', '~spam'])
ax.set_yticklabels(['spam', '~	spam'])	
plt.show()