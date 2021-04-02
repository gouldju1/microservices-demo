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

#Map Norma PA and DP
def dp_and_pa_approach_b_map(dp_mapped, polarities):
    for part in dp_mapped:
        causal_parts = []
        #Get List of Verbs for Part
        verbs = dp_mapped[part]
        for verb in verbs:
            if verb in polarities["negative_words"]:
                causal_parts.append(part)

    return causal_parts