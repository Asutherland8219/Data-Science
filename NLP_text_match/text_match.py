from urllib import response
import spacy 
from spacy.matcher import PhraseMatcher
from textblob import Sentence
import csv 
import json
import numpy as np

nlp = spacy.load('en_core_web_md')
phrase_matcher = PhraseMatcher(nlp.vocab)


# this is where you would add the tickers
with open("NLP_text_match\sp_500_stocks.json", "r") as read_file:
    data = json.load(read_file)
    
    data2 = json.loads(str(data))

print(data2)

# for i in data:
#     response_set = []
#     i_set = i
#     response_set.append(i_set)
#     for x in response_set:
#         y = np.array(x)
       
# phrases = data
# patterns = [nlp(text) for text in phrases]


# phrase_matcher.add('AI', None, *patterns)

# the html of the link
#processed_article = html



# sentence = nlp(processed_article)
# matched_phrases = phrase_matcher(sentence)

# for match_id, start, end in matched_phrases:
#     string_id = nlp.vocab.strings[match_id]
#     span = sentence[start:end]
#     print(match_id, string_id, start, end, span.text)
    


