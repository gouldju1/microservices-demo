#Required Packages
import spacy
import os
import re

#Load Model
dp_model = spacy.load('en_core_web_sm')

#Run Model
def dp(params):
    #Unpack Parameters
    text = params["text"]
    if "deps" in params:
        deps = params["deps"]
    else:
        deps = ["amod", "ccomp", "acl", "nsubj"]
    
    #Run Model
    doc = dp_model(text)
    
    data_dp = {}
    for token in doc:
        if token.dep_ in ["amod", "ccomp", "acl", "nsubj"]:
            noun = str(token.head)
            verb = str(token.text)
            
            #Block to append to dictionary------
            #Existing label append to list...
            if noun in data_dp:
                data_dp[noun].append(verb)

            #If label does not exist, create
            else:
                data_dp[noun] = [verb]
            
    #Remove Duplicated
    for key in data_dp:
        data_dp[key] = list(set(data_dp[key]))
                
    return data_dp