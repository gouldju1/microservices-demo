#Required Packages
import sys
import requests
import json

#System Path
sys.path.append("../")

#NER Model
from src import approach_a, approach_b, utils

#Approach A
def approach_a_pipeline(sample):
    #NER
    payload = {
    "text" : sample
    }

    url = "http://127.0.0.1:5000/ner"
    r = requests.post(url, data=json.dumps(payload), headers={"content-type":"application/json; charset=utf-8"})
    resp = json.loads(r.content.decode("utf-8"))
    entities = resp["output"]
    
    #DP
    payload = {
    "text" : sample,
    "deps" : ["amod", "ccomp", "acl", "nsubj"]
    }

    model_to_use = "dep_students"
    url = f"http://127.0.0.1:5000/{model_to_use}"
    r = requests.post(url, data=json.dumps(payload), headers={"content-type":"application/json; charset=utf-8"})
    resp = json.loads(r.content.decode("utf-8"))
    dp_dict = resp["output"]

    #Map DP to NER
    payload = {
    "dp_dict"  : dp_dict,
    "entities" : entities
    }

    url = "http://127.0.0.1:5000/map_dp_ner"
    r = requests.post(url, data=json.dumps(payload), headers={"content-type":"application/json; charset=utf-8"})
    resp = json.loads(r.content.decode("utf-8"))
    dp_mapped = resp["output"]

    #PA
    params = {
    "input_dict" : dp_mapped
    }

    approach_a_polarity = approach_a.predict_polarity(params)
    causal_part_candidates_approach_a = approach_a_polarity["non_positive_parts"]
    
    return causal_part_candidates_approach_a

#Approach B
def approach_b_pipeline(sample):
    #Lowercase for PA
    sample = sample.lower()
    
    #NER
    payload = {
    "text" : sample
    }

    url = "http://127.0.0.1:5000/ner"
    r = requests.post(url, data=json.dumps(payload), headers={"content-type":"application/json; charset=utf-8"})
    resp = json.loads(r.content.decode("utf-8"))
    entities = resp["output"]
    
    #PA
    params = {
    "text" : sample
    }

    polarity_approach_b = approach_b.approach_b_polarity(params)
    
    #DP
    payload = {
    "text" : sample,
    "deps" : ["amod", "ccomp", "acl", "nsubj"]
    }

    model_to_use = "dep_students"
    url = f"http://127.0.0.1:5000/{model_to_use}"
    r = requests.post(url, data=json.dumps(payload), headers={"content-type":"application/json; charset=utf-8"})
    resp = json.loads(r.content.decode("utf-8"))
    dp_dict = resp["output"]
    
    #Map DP to NER
    payload = {
    "dp_dict"  : dp_dict,
    "entities" : entities
    }

    url = "http://127.0.0.1:5000/map_dp_ner"
    r = requests.post(url, data=json.dumps(payload), headers={"content-type":"application/json; charset=utf-8"})
    resp = json.loads(r.content.decode("utf-8"))
    dp_mapped = resp["output"]
    
    #DP Mapped to PA
    causal_part_candidates_approach_b = utils.dp_and_pa_approach_b_map(dp_mapped, polarity_approach_b)
    
    return causal_part_candidates_approach_b