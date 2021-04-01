import torch
import numpy as np
from transformers import BertTokenizer, BertModel
import torch.nn as nn
import torch.optim as optim


# replace 'path' with the path to the location to the 10epochRNN.pt


model_save_name = 'polarity_model.pt'
path = F"/Users/gould29/OneDrive - purdue.edu/Purdue/The Data Mine/AY 2020-2021/Cummins/{model_save_name}"


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

init_token = tokenizer.cls_token
eos_token = tokenizer.sep_token
pad_token = tokenizer.pad_token
unk_token = tokenizer.unk_token
max_input_length = tokenizer.max_model_input_sizes['bert-base-uncased']

init_token_idx = tokenizer.cls_token_id
eos_token_idx = tokenizer.sep_token_id
pad_token_idx = tokenizer.pad_token_id
unk_token_idx = tokenizer.unk_token_id

#instead of SpaCy, handles all tokenization
def tokenize_and_cut(sentence):
    tokens = tokenizer.tokenize(sentence) 
    tokens = tokens[:max_input_length-2]
    return tokens


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

bert = BertModel.from_pretrained('bert-base-uncased')

class BERTRNNPolarity(nn.Module):
    def __init__(self,
                 bert,
                 hidden_dim,
                 output_dim,
                 n_layers,
                 bidirectional,
                 dropout):
        
        super().__init__()
        
        self.bert = bert
        
        embedding_dim = bert.config.to_dict()['hidden_size']
        
        self.rnn = nn.RNN(embedding_dim,
                          hidden_dim,
                          num_layers = n_layers,
                          bidirectional = bidirectional,
                          nonlinearity  = 'relu',
                          batch_first = True,
                          dropout = 0 if n_layers < 2 else dropout)
        
        self.out = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        
        #text = [batch size, sent len]
                
        with torch.no_grad():
            embedded = self.bert(text)[0]
                
        #embedded = [batch size, sent len, emb dim]
        
        _, hidden = self.rnn(embedded)
        
        #hidden = [n layers * n directions, batch size, emb dim]
        
        if self.rnn.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        else:
            hidden = self.dropout(hidden[-1,:,:])
                
        #hidden = [batch size, hid dim]
        
        output = self.out(hidden)
        
        #output = [batch size, out dim]
        
        return output

HIDDEN_DIM = 256
OUTPUT_DIM = 1
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.25

model = BERTRNNPolarity(bert,
                         HIDDEN_DIM,
                         OUTPUT_DIM,
                         N_LAYERS,
                         BIDIRECTIONAL,
                         DROPOUT)
model.load_state_dict(torch.load(path, device))
model.eval()


optimizer = optim.Adam(model.parameters())
criterion = nn.BCEWithLogitsLoss()
model = model.to(device)
criterion = criterion.to(device)


# def predict_polarity(params):
    
#     #unpack Parameters
#     words = list(params.keys())
#     sentences = list(params.values())   
    
#     #model.eval()
#     polarity_dict = {}
       
#     for word in words:
#         tokens = tokenizer.tokenize(word)
#         tokens = tokens[:max_input_length-2]
#         indexed = [init_token_idx] + tokenizer.convert_tokens_to_ids(tokens) + [eos_token_idx]
#         tensor = torch.LongTensor(indexed).to(device)
#         tensor = tensor.unsqueeze(0)
#         prediction = torch.sigmoid(model(tensor))
#         if prediction.item() >= 0.5:
#             polarity_dict[word] = "Positive"
#         else:
#             polarity_dict[word] = "Negative"
       
#     return polarity_dict

def predict_polarity(params):
    
    #Unpack Parameters
    pad = params["input_dict"]
    
    #Final Dictionary for Output
    output_dict = {}
    
    #List of non-positive parts as causal
    non_positive_parts = []
    
    #Dictionary to store polarity information
    polarity_dict = {}
    
    #Loop through Matched DP-NER
    for key in pad:
        #Temporary, word-level dict
        internal_dict = {}
        
        #List of verbs per part
        verbs = pad[key]
        
        #Loop through Tokens Per Part
        for word in verbs:
            #Tokenize and run prediction
            tokens = [word][:max_input_length-2]
            indexed = [init_token_idx] + tokenizer.convert_tokens_to_ids(tokens) + [eos_token_idx]
            tensor = torch.LongTensor(indexed).to(device)
            tensor = tensor.unsqueeze(0)
            prediction = torch.sigmoid(model(tensor))
            
            #Interpret Prediction
            if prediction.item() > 0.5:
                polarity = "Positive"
            else:
                polarity = "Non-positive"
                #If non-positive, add to list
                non_positive_parts.append(key)
            
            #Compile output data
            if key in polarity_dict:
                internal_dict[word] = polarity
            else:
                internal_dict[word] = polarity
                polarity_dict[key] = internal_dict
                
    #Assemble Final Dictionary
    output_dict = {
        "non_positive_parts" : list(set(non_positive_parts)), #Remove duplicates
        "polarity" :           polarity_dict
    }
       
    return output_dict
