''' Create a model that summarizes documents'''
from typing import Text
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
import re 
from io import StringIO

# For extraction and feature modeling
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction._stop_words import ENGLISH_STOP_WORDS

# other
import numpy as np 
import pandas as pd 

''' Data prep '''
def convert_pdf_to_txt(path):
    rsrcmgr = PDFResourceManager()
    retstr = StringIO()
    laparams = LAParams()
    device = TextConverter(rsrcmgr, retstr, laparams=laparams)
    fp = open(path, 'rb')
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    password = ""
    maxpages = 0
    caching = True
    pagenos = set()

    for page in PDFPage.get_pages(fp, pagenos, maxpages=maxpages, password=password, caching = caching, check_extractable = True ):
        interpreter.process_page(page)


    text = retstr.getvalue()

    fp.close()
    device.close()
    retstr.close()
    return text


# pick a random 10k and just fill that in ; i used AAPL. 
Document = convert_pdf_to_txt('Workbook_ex\Datasets\AAPL-10k.pdf')
f = open('Workbook_ex\Datasets\Finance10k.txt', 'w', encoding='utf-8')
f.write(Document)
f.close()
with open('Workbook_ex\Datasets\Finance10k.txt', encoding='utf-8') as f:
    clean_cont = f.read().splitlines()

print(clean_cont[1:15])
# remove the filler and spaces 
doc = [i.replace('\xe2\x80\x9c', '') for i in clean_cont ]
doc = [i.replace('\xe2\x80\x9d', '') for i in doc]
doc = [i.replace('\xe2\x80\x99s', '') for i in doc]

docs = [x for x in doc if x != ' ']
docss = [x for x in docs if x!=  '']
financedoc = [re.sub("[^a-zA-Z]+", " ", s) for s in docss]

print(financedoc[1:30])

''' Model Construction and training '''
# use the CountVectorizer method to show occurance of words and create topics for the model 
vect = CountVectorizer(ngram_range=(1,1), stop_words='english')
fin = vect.fit_transform(financedoc)
pd.DataFrame(fin.toarray(), columns=vect.get_feature_names()).head(1)

# furthermore we use the Latent Dirichlet Allocation Algorithm for the modeling (https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation)
lda = LatentDirichletAllocation(n_components=5)
lda.fit_transform(fin)
lda_dtf = lda.fit_transform(fin)
sorting = np.argsort(lda.components_)[:, ::-1]
features = np.array(vect.get_feature_names())

array = np.full((1, sorting.shape[1]), 1)
array = np.concatenate((array,sorting), axis= 0)

import mglearn
topics = mglearn.tools.print_topics(topics=range(1,6), feature_names=features, sorting=array, topics_per_chunk=5, n_words=10)

''' Visualize '''
import pyLDAvis
import pyLDAvis.sklearn

zit = pyLDAvis.sklearn.prepare(lda,fin,vect)
pyLDAvis.display(zit)



''' Word cloud '''
from PIL import Image
import numpy as np
import matplotlib.pyplot as pyplot
from wordcloud import WordCloud, STOPWORDS

text = open('Workbook_ex\Datasets\Finance10k.txt', encoding='utf-8').read()
stopwords = set(STOPWORDS)
wc = WordCloud(background_color='black', max_words=2000, stopwords=stopwords)
wc.generate(text)

pyplot.figure(figsize=(16,13))
pyplot.imshow(wc, interpolation='bilinear')
pyplot.axis('off')
pyplot.figure()

pyplot.axis('off')
pyplot.show()


