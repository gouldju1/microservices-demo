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

#run model
def dep_students(params):
    """
    Description:
        Wrapper to run the Purdue Digital Crossroads Dependency Parsing model on Cummins service text.
        
    Input:
     - text (str):        Input service text, as decoded (utf-8) text
    
    Output:
        {
            part_name: [list, of, extracted, causes]
            part_name: [list, of, extracted, causes]
            ...
        }
    """
    
    #Unpack Parameters
    text = params["text"]
    
    #Run model
    doc = dp_model(text)
    
    dep_list = []
    
    for token in doc:
        if token.dep_ == "amod":
            single_dep = ()
            if token.head.pos_ == "NOUN" or token.head.pos_ == "PROPN":
                single_dep = (str(token.head), token.text)
                dep_list.append(single_dep)
        
            elif token.pos_ == "NOUN" or token.pos_ == "PROPN":
                single_dep = (token.text, str(token.head))
                dep_list.append(single_dep)
            #single_dep = part_first(token)
        
        
        elif token.dep_ == "dobj":
            single_dep = ()
            if token.head.pos_ == "NOUN" or token.head.pos_ == "PROPN":
                single_dep = (str(token.head), token.text)
                dep_list.append(single_dep)
        
            elif token.pos_ == "NOUN" or token.pos_ == "PROPN":
                single_dep = (token.text, str(token.head))
                dep_list.append(single_dep)
            #single_dep = part_first(token)
            
        elif token.dep_ == "nsubj":
            single_dep = ()
            if token.head.pos_ == "NOUN" or token.head.pos_ == "PROPN":
                single_dep = (str(token.head), token.text)
                dep_list.append(single_dep)
        
            elif token.pos_ == "NOUN" or token.pos_ == "PROPN":
                single_dep = (token.text, str(token.head))
                dep_list.append(single_dep)
            #single_dep = part_first(token)
        #print(dep_list)
    
    dep_dict = {}

    for i in dep_list:
        dep_dict.setdefault(i[0],[]).append(i[1])
    
    return dep_dict