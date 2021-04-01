#Required Packages
import nltk
import random
from spacy.lang.en import English
import dill as pickle
import os
import sys

#Internal Packages
sys.path.append("../")
from src import Subjectivity

#Set up Tokenizer
nlp = English()
tokenizer = nlp.tokenizer

#Dependencies Path
base_path = "/Users/gould29/OneDrive - purdue.edu/Purdue/The Data Mine/AY 2020-2021/Cummins/NLP Models"

#Set up SL
SL_name = "subjclueslen1-HLTEMNLP05.tff"
SL = Subjectivity.readSubjectivity(os.path.join(base_path, SL_name))

#Load NLTK Model
model_name = "polarity.pk"
with open(os.path.join(base_path, model_name), 'rb') as fin:
    polarity_nltk = pickle.load(fin)

#Create SL Feature Set
def SL_features(document, word_features, SL):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['V_{}'.format(word)] = (word in document_words)
    # count variables for the 4 classes of subjectivity
    weakPos = 0
    strongPos = 0
    weakNeg = 0
    strongNeg = 0
    for word in document_words:
        if word in SL:
            strength, posTag, isStemmed, polarity = SL[word]
            if strength == 'weaksubj' and polarity == 'positive':
                weakPos += 1
            if strength == 'strongsubj' and polarity == 'positive':
                strongPos += 1
            if strength == 'weaksubj' and polarity == 'negative':
                weakNeg += 1
            if strength == 'strongsubj' and polarity == 'negative':
                strongNeg += 1
            features['positivecount'] = weakPos + (2 * strongPos)
            features['negativecount'] = weakNeg + (2 * strongNeg)      
    return features

#Prepare for Model Inference
def prep_inference(featuresets):
    for fs in featuresets:
        internal = []
        for key in fs:
            internal.append({key : fs[key]})
    
    return internal

#Post-model Output Formatting
def output_formatting(preds, internal):
    output = {}
    negatives = []
    final = {}
    for pred, tok in zip(preds, internal):
        word = list(tok.keys())[0].replace("V_", "")
        output[word] = pred
        if pred == "neg":
            negatives.append(word)
    
    #Assemble final output
    final = {
        "negative_words" : negatives,
        "polarities"     : output
    }

    return final

    
#Final Pipeline
def norma_polarity(params):
    #Unpack Parameters
    text = params["text"].lower()
    
    #Tokenize Document
    doc = tokenizer(text)
    tokens = [word.text for word in doc]
    
    #Create Frequency Distribution
    all_words = nltk.FreqDist(tokens)
    word_items = all_words.most_common(5000)
    word_features = [word for (word, count) in word_items]
    
    #SL Fecture Set
    SL_featuresets = [SL_features(tokens, word_features, SL)]
    
    #Prep Feature for Inference
    internal = prep_inference(SL_featuresets)[:-2]
    
    #Run Model
    preds = polarity_nltk.classify_many(internal)
    
    #Map Predictions and Document Tokens
    output = output_formatting(preds, internal)
    
    return output