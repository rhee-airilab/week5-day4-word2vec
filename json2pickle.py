# coding: utf-8
import json
import pickle
from pprint import pprint

#filename = '../namu/namuwiki_170327.json' 
filename = 'namuwiki_170327.json' 

# Read file to memory, it takes some time.
with open(filename) as data_file:    
    data = json.load(data_file)

with open('namuwiki.pickle','wb') as pickle_file:
    pickle.dump(data,pickle_file)
   
