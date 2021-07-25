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

### you may run into the time.clock() does not exist issue; simply navigate to C/Anaconda3/envs/(env name)/lib/site-packages/sqlalchemy/util/compat then open the file in word doc and remove the time.clock() try function

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
# # def converse(quit= "quit"):
#     user_input = ""
#     while user_input != quit:
#         user_input = quit
#         try:
#             user_input = input(">")
#         except EOFError:
#             print(user_input)
#         if user_input:
#             while user_input[-1] in "!.":
#                 user_input = user_input[:-1]
#             print(chatdefault.get_response(user_input))

''' Now we can make a custom bot that returns financial data'''
companies = {
    'AAPL': ['Apple', 'Apple Inc'],
    'BAC': ['BAML', 'BofA', 'Bank of America'],
    'C': ['Citi', 'Citibank'],
    'DAL': ['Delta', 'Delta Airlines']
}

ratios = {
    'return-on-equity-ttm': ['ROE', 'Return on Equity'],
    'cash-from-operations-quarterly':['CFO', 'Cash Flow from Operations'],
    'pe-ratio-ttm': ['PE', 'Price to equity', 'pe ratio'],
    'revenue-ttm': ['Sales', 'Revenue']
}

string_templates = [
    'Get me the {ratio} for {company}',
    'What is the {ratio} for {company}?',
    'Tell me the {ratio} for {company}',
    '{ratio} for {company}'
]

''' Construct the new mmodel '''
companies_rev = {}
for k, v in companies.items():
    for ve in v:
        companies_rev[ve] = k
    ratios_rev = {}
    for k, v in ratios.items():
        for ve in v:
            ratios_rev[ve] = k
    companies_list = list(companies_rev.keys())
    ratios_list = list(ratios_rev.keys())

# Get the training data 
import random
N_training_samples = 100
def get_training_sample(string_templates, ratios_list, companies_list):
    string_template= string_templates[random.randint(0, len(string_templates)-1)]
    ratio = ratios_list[random.randint(0, len(ratios_list)-1)]
    company = companies_list[random.randint(0, len(companies_list)-1)]
    sent = string_template.format(ratio=ratio, company = company)
    ents = {"entities": [(sent.index(ratio), sent.index(ratio) + len(ratio), 'RATIO'),
    (sent.index(company), sent.index(company) + len(company), 'COMPANY')]}

    return (sent, ents)

TRAIN_DATA = [
    get_training_sample(string_templates, ratios_list, companies_list)
    for i in range(N_training_samples)
]

# Initialize the model 

nlp = spacy.blank('en')

ner = nlp.create_pipe('ner')
nlp.add_pipe('ner')

ner.add_label('RATIO')
ner.add_label('COMPANY')

''' Model optimiztaion function '''

optimizer = nlp.begin_training()
move_names = list(ner.move_names)
pipe_exceptions = ['ner', 'trf_wordpiecer', 'trf_tok2vec']
other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
with nlp.disable_pipes(*other_pipes):
    sizes = compounding(1.0, 4.0, 1.001)
    for itn in range(30):
        random.shuffle(TRAIN_DATA)
        batches = minibatch(TRAIN_DATA, size= sizes)
        losses = {}
    for batch in batches:
        texts, annotations = zip(*batch)
        nlp.update(texts, annotations, sgd= optimizer,
        drop= 0.35, losses=losses)
    print("Losses", losses)

''' Custom logic gate '''
from chatterbot.conversation import Statement
class FinancialRatioAdapter(LogicAdapter):
    def __init__(self, chatbot, **kwargs):
        super(FinancialRatioAdapter, self).__init__(chatbot, **kwargs)
    def process(self, statement, additional_response_selection_parameters):
        user_input = statement.text
        doc = nlp(user_input)
        company = None
        ratio = None
        confidence = 0
        # neeed 1 company and 1 ratio 
        if len(doc.ents) == 2:
            for ent in doc.ents:
                if ent.label_ == "RATIO":
                    ratio = ent.text
                    if ratio in ratios_rev:
                        confidence += 0.5
                if ent.label_ == "COMPANY":
                    company = ent.text
                    if company in companies_rev:
                        confidence += 0.5
        if confidence > 0.99: #(found ratio and company)
            outtext = '''https://www.zacks.com/stock/chart/{company}/fundamental{ratio}'''.format(ratio=ratios_rev[ratio], company=companies_rev[company])
            confidence = 1 
        else:
            outtext = 'Sorry! Could not figure out what user wants'
            confidence = 0 
        output_statement = Statement(text=outtext)
        output_statement.confidence = confidence
        return output_statement

from chatterbot import ChatBot

chatbot= ChatBot(
    "My Chatterbot",
    logic_adapters=[
        'financial_ratio_adapter.FinancialRatioAdapter'
    ]
)

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



