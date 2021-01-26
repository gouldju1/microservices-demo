#Required Packages
from nltk import sent_tokenize
from spacy.lang.en import English
nlp = English()
tokenizer = nlp.Defaults.create_tokenizer(nlp)

#Split and Tokenize
def split_and_tokenize(text):
    """
    Goal: Create a list of list of tokens from sentence(s).
    Input:
      - text: String containing >= 1 sentence
    Output:
      - List of list of tokens from sentence(s).
    """
    
    #Split Sentences
    sentences = sent_tokenize(text)

    #Tokenize Each Sentence
    all_tokens = []
    for sentence in sentences:
        #Internal List for Each Sentence
        tokens = []

        #Tokenize Sentence
        tokenized = tokenizer(sentence)
        for token in tokenized:
            #Add to Internal List
            tokens.append(str(token))

        #Add Internal to Overall List
        all_tokens.append(tokens)
        
    return all_tokens

#Tokenize
def tokenize(sentence):
    """
    Goal: Tokenize input text.
    Input:
      - sentence: String of text to tokenize via spacy
    Output:
      - List of tokens from input text
    """
    
    #Token List
    tokens = []
    
    #Tokenize
    tokenized = tokenizer(sentence)
    for token in tokenized:
        tokens.append(str(token))
    
    return tokens

#Split Sentences
def split_sentences(text):
    """
    Goal: Split sentences from input text.
    Input:
      - text: String containing >= 1 sentence.
    Output:
      - List of sentence(s).
    """
    
    #Split Sentences
    sentences = sent_tokenize(text)
    
    return sentences