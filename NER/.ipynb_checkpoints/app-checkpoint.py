#Required Packages
import json
from flask import Flask, jsonify, request
import sys

#Handle Internal Packages
package_path = "./"
if package_path not in sys.path:
    sys.path.append(package_path)
    
#Internal Package(s)
from src import ner, dependency_parsing, utils#, norma

#Establish App
app = Flask(__name__)

#Functions
#Default
@app.route('/')
def index():
    return jsonify({'name': 'Justin Gould',
                    'email': 'gould29@purdue.edu'})

#NER
@app.route('/ner', methods=["POST"])
def ner_model():
    #Run Function
    p = request.get_json()
    output = ner.ner(p)
    return jsonify({'output': output})

#DP
@app.route('/dp', methods=["POST"])
def dp_model():
    #Run Function
    p = request.get_json()
    output = dependency_parsing.dp(p)
    return jsonify({'output': output})

#Map DP and NER
@app.route('/map_dp_ner', methods=["POST"])
def map_dp_ner():
    #Run Function
    p = request.get_json()
    output = utils.map_dp_and_ner(p)
    return jsonify({'output': output})

# #Polarity
# @app.route('/norma_polarity', methods=["POST"])
# def norma_polarity_model():
#     #Run Function
#     p = request.get_json()
#     output = norma.norma_polarity(p)
#     return jsonify({'output': output})

#Run Application
app.run()