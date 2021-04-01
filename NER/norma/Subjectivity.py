# Module Subjectivity reads the subjectivity lexicon file from Wiebe et al
#    at http://www.cs.pitt.edu/mpqa/ (part of the Multiple Perspective QA project)
#
# This file has the format that each line is formatted as in this example for the word "abandoned"
#     type=weaksubj len=1 word1=abandoned pos1=adj stemmed1=n priorpolarity=negative
# In our data, the pos tag is ignored, so this program just takes the last one read
#     (typically the noun over the adjective)
#
# The data structure that is created is a dictionary where
#    each word is mapped to a list of 4 things:  
#        strength, which will be either 'strongsubj' or 'weaksubj'
#        posTag, either 'adj', 'verb', 'noun', 'adverb', 'anypos'
#        isStemmed, either true or false
#        polarity, either 'positive', 'negative', or 'neutral'

import nltk

# pass the absolute path of the lexicon file to this program
# example call:
# nancymacpath = 
#    "/Users/njmccrac/AAAdocs/research/subjectivitylexicon/hltemnlp05clues/subjclueslen1-HLTEMNLP05.tff"
# SL = readSubjectivity(nancymacpath)

# this function returns a dictionary where you can look up words and get back 
#     the four items of subjectivity information described above
def readSubjectivity(path):
    flexicon = open(path, 'r')
    # initialize an empty dictionary
    sldict = { }
    for line in flexicon:
        fields = line.split()   # default is to split on whitespace
        # split each field on the '=' and keep the second part as the value
        strength = fields[0].split("=")[1]
        word = fields[2].split("=")[1]
        posTag = fields[3].split("=")[1]
        stemmed = fields[4].split("=")[1]
        polarity = fields[5].split("=")[1]
        if (stemmed == 'y'):
            isStemmed = True
        else:
            isStemmed = False
        # put a dictionary entry with the word as the keyword
        #     and a list of the other values
        sldict[word] = [strength, posTag, isStemmed, polarity]
    return sldict