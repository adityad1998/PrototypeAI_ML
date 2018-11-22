from __future__ import absolute_import
from __future__ import print_function
import six

import train
import operator
import io

stoppath = "data/excludelist.txt"
rake_object = train.Rake(stoppath, 5, 3, 4)
sample_file = io.open("data/testset/w2167e.txt", 'r',encoding="iso-8859-1")
text = sample_file.read()
keywords = rake_object.run(text)
#print("Keywords:", keywords)
rake_object = train.Rake(stoppath)

text = "Behrendorff to Dhawan, no run, he charges, banged in back of a length, he gives room and cuts straight to backward point. In machine learning, support vector machines are supervised learning models with associated learning algorithms that analyze data used for classification and regression analysis."

sentenceList = train.split_sentences(text)
stopwords = train.load_stop_words(stoppath)
stopwordpattern = train.build_stop_word_regex(stoppath)
phraseList = train.generate_candidate_keywords(sentenceList, stopwordpattern, stopwords)
#print("Phrases:", phraseList)
wordscores = train.calculate_word_scores(phraseList)
keywordcandidates = train.generate_candidate_keyword_scores(phraseList, wordscores)
#for candidate in keywordcandidates.keys():
    #print("Candidate: ", candidate, ", score: ", keywordcandidates.get(candidate))
sortedKeywords = sorted(six.iteritems(keywordcandidates), key=operator.itemgetter(1), reverse=True)
totalKeywords = len(sortedKeywords)
#for keyword in sortedKeywords[0:int(totalKeywords / 3)]:
    #print("Keyword: ", keyword[0], ", score: ", keyword[1])
print(rake_object.run(text))
