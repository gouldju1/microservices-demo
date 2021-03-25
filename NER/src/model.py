#Required Packages
import spacy
import os
import re

#Load Model    
base = "/Users/gould29/OneDrive - purdue.edu/Purdue/The Data Mine/AY 2020-2021/Cummins"
model = "tmp_model_14"
model_path = os.path.join(base, model)
nlp = spacy.load(model_path)

#Run Model
def ner(params):
    """
    Description:
        Wrapper to run the Purdue Digital Crossroads NER model on Cummins service text.
    
    Input, dictionary containing:
     - text (str):             Input service text, as decoded (utf-8) text
     - part_num_to_int (bool): If a part number is extracted, should it remain a str or convert to int?
    
    Output:
        {
            entity_name: [list, of, extracted, entities],
            entity_name: [list, of, extracted, entities],
            ...
        }
    """
    
    #Unpack Parameters
    text = params["text"]
    if "part_num_to_int" in params:
        part_num_to_int = params["part_num_to_int"]
    else:
        part_num_to_int = False
    
    #Run Model
    doc = nlp(text)
    
    #Parse------
    data = {}
    for ent in doc.ents:
        #Preprocess entity text------
        ent_text = ent.text
        #LABEL in TEXT
        ent_text = ent_text.replace(ent.label_, "")

        #For PART_NUMBER, extract only the numbers
        if ent.label_ == "PART_NUM":
            ent_text = "".join(re.findall(r'\d+', ent_text))
            if part_num_to_int:
                ent_text = int(ent_text)

        #Block to append to dictionary------
        #Existing label append to list...
        if ent.label_ in data:
            data[ent.label_].append(ent_text)

        #If label does not exist, create
        else:
            data[ent.label_] = [ent_text]
    
    return data