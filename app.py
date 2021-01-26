#Required Packages
import json
from flask import Flask, jsonify, request
import sys

#Handle Internal Packages
package_path = "./src/"
if package_path not in sys.path:
    sys.path.append(package_path)
    
#Internal Package(s)
from utils import example

#Establish App
app = Flask(__name__)

#Functions
#Default
@app.route('/')
def index():
    return jsonify({'name': 'Justin Gould',
                    'email': 'gould29@purdue.edu'})

#Combined Split and Tokenize
@app.route('/split_and_tokenize', methods=["POST"])
def split_and_tokenize():
    #Run Function
    p = request.get_json()
    print(p, "\n", p["text"], "\n")
    
    output = example.split_and_tokenize(p["text"])
    print(output)
    return jsonify({'output': output})

#Split Sentences
@app.route('/split_sentences', methods=["POST"])
def split_sentences():
    #Run Function
    p = request.get_json()
    output = example.split_sentences(p["text"])
    return jsonify({'output': output})

#Tokenize a Sentence
@app.route('/tokenize', methods=["POST"])
def tokenize():
    #Run Function
    p = request.get_json()
    output = example.tokenize(p["sentence"])
    return jsonify({'output': output})

#Run Application
app.run()