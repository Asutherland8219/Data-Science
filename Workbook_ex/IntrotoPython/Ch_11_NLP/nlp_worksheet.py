# This is a collection of work snippets from the chapter 
from typing import NoReturn, Text
from textblob import TextBlob

"""Textblob"""
text = 'Today is a beautiful day. Tomorrow looks like bad weather.'

blob = TextBlob(text)

print(blob.__dict__)


"""Tokenizing"""
# sentences feature breaks apart sentences via the period
print(blob.sentences)

# Words breaks down each word by spaces 
print(blob.words)


"""Parts of Speech tagging"""
# tag each word with their word attributes like verb, noun etc...
print(blob.tags)

# Here is the index for each word (note this is just a sample, there are 63 total):
#  NN is singular or mass noun`
#  VBZ is third person singular verb
#  DT is determiner
#  JJ is adjective
#  NNP is proper singular noun
#  IN is a subordinating conjuction or preposition
# Full list at www.clips.uantwerpen.be/pages/MBSP-tags

# Noun phrases ; nouns vs the following/previous word
print(blob.noun_phrases)

"""Sentiment Analysis"""
# Polarity is -1 to 1 with 0 being neutral
# Subjectivities is 0 (objective) to 1 (subjective)
print(blob.sentiment)

# Can also have sentence sentiment
for sentence in blob.sentences:
    print(sentence.sentiment)

# expanding on this further we can use NaiveBayes models

"""Sentiment analysis with NaiveBayesAnalyzer"""
from textblob.sentiments import NaiveBayesAnalyzer

blob = TextBlob(text, analyzer=NaiveBayesAnalyzer())

print(blob.sentiment)
# the naive bayes analyzer adds a classfication (positive, negative etc..)
# also applies to sentences
for sentence in blob.sentences:
     print(sentence.sentiment)

"""Language Detection"""
print(blob.detect_language)

spanish = blob.translate(to='es')
print(spanish)
chinese = blob.translate(to='zh')
print(chinese)
print(chinese.detect_language())

"""Inflection: Pluralization and Singularization"""
from textblob import Word

# take a singular word and make it plural
index = Word('index')
print(index.pluralize())

# take a plural word and singularize it
cacti = Word('cacti')
print(cacti.singularize())

# Spellcheck also exists
word = Word('tel')
print(word.spellcheck()) #spell check matches based on % value what it could be
print(word.correct()) # correct picks the highest value in spell check

"""Normalization: Stemming and Lemmatization"""

# stemming removes prefix or suffix
word = Word('varieties')
print(word.stem())

# lemmatizing removes the prefix or suffix but ensures a word is made 
print(word.lemmatize())

""" Word Frequency """
# count the frequency of words in a blob of teext 
# see page 492 in Intry to python book.add()

"""Getting definitions, synonyms, antonyms"""
# Princeton has a database of words with definitions
hidden = Word('Sequestered')
print(hidden.definitions)

# Synonyms
print(hidden.synsets)

# Antonyms
lemmas = hidden.synsets[0].lemmas()
print(lemmas)

antonym= lemmas[0].antonyms()
print(antonym)

""" Using Stop words """
# we can download stopwords
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
stops= stopwords.words('english')

print([word for word in blob.words if word not in stops])

"""" n-grams """
# ngrams arfe a sequence of n text items
# the default is 3 but you can specify how many you would like

print(blob.ngrams())
print(blob.ngrams(5))









