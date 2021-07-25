''' Create Chatbot '''
import pkg_resources
import pip 
installedPackages = {pkg.key for pkg in pkg_resources.working_set}
import spacy
from spacy.util import minibatch, compounding


# Load chatterbot 
from chatterbot import ChatBot
from chatterbot.logic import LogicAdapter, logic_adapter
from chatterbot.trainers import ChatterBotCorpusTrainer
from chatterbot.trainers import ListTrainer

#disable warnings 
import warnings
warnings.filterwarnings('ignore')

''' Train the default chatbot '''
chatdefault = ChatBot("Trader", preprocessors=['chatterbot.preprocessors.clean_whitespace'], logic_adapters= ['chatterbot.logic.BestMatch', 'chatterbot.logic.MathematicalEvaluation'
])

# corpus training 
trainerCorpus = ChatterBotCorpusTrainer(chatdefault)

# train in English
trainerCorpus.train("chatterbot.corpus.english")

# train based on standard english greetings 
trainerCorpus.train("chatterbot.corpus.english.conversations")

trainerConversation = ListTrainer(chatdefault)


# the test to train it on 
trainerConversation.train([
    'Help!',
    'Please go to google.com',
    'What is Bitcoin?',
    'It is a decentralized digital currency'
])

# 2nd list of data for response variations
trainerConversation.train([
    'What is Bitcoin?',
    'Bitcoin is a cryptocurrency.'
])
 
''' Create a function to initialize '''
def converse(quit= "quit"):
    user_input = ""
    while user_input != quit:
        user_input = quit
        try:
            user_input = input(">")
        except EOFError:
            print(user_input)
        if user_input:
            while user_input[-1] in "!.":
                user_input = user_input[:-1]
            print(chatdefault.get_response(user_input))
