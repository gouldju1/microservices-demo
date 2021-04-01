#Required Packages


#Map DP to NER
def map_dp_and_ner(params):
    #Unpack Parameters
    dp_dict = params["dp_dict"]
    entities = params["entities"]
    
    data = {}
    #Loop through DP Output
    for dp_part in dp_dict:
        #Loop through NER Parts Extracted
        for part in entities["PART_NAME"]:
            part_in_dp_and_ner = ((dp_part in part) | (part in dp_part))

            if part_in_dp_and_ner is True:
                data[part] = dp_dict[dp_part]
        
    return data