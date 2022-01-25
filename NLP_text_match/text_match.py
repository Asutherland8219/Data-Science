import spacy 
from spacy.matcher import PhraseMatcher
from textblob import Sentence

nlp = spacy.load('en_core_web_sm')
phrase_matcher = PhraseMatcher(nlp.vocab)


# this is where you would add the tickers
phrases = []
patterns = [nlp(text) for text in phrases]


phrase_matcher.add('AI', None, *patterns)

# the html of the link
#processed_article = html

sentence = nlp(processed_article)
matched_phrases = phrase_matcher(sentence)

for match_id, start, end in matched_phrases:
    string_id = nlp.vocab.strings[match_id]
    span = sentence[start:end]
    print(match_id, string_id, start, end, span.text)
    


