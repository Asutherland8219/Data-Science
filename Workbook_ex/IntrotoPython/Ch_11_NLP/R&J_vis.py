from pathlib import Path
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
import pandas as pd

blob = TextBlob(Path('Workbook_ex\IntrotoPython\Ch_11_NLP\R&J_prepped.txt').read_text())

# make sure to download stopwords
nltk.download('stopwords')
stop_words = stopwords.words('english')

# Getting the word frequencies
items = blob.word_counts.items()

# elminating the stop words
items = [item for item in items if item[0] not in stop_words]

# sort via frequency
from operator import itemgetter
sorted_items = sorted(items, key=itemgetter(1), reverse=True)

# count the top 20 words 
top20 = sorted_items[1:21]

# Now we use pandas to make a dataframe 
df = pd.DataFrame(top20, columns=['word', 'count'])

print(df)

""" Visualize the results """
import matplotlib.pyplot as pyplot
pyplot.gcf().tight_layout()
axes = df.plot.bar(x='word', y='count', legend=False)

import imageio
mask_image = imageio.imread('mask_heart.png')

# configuring the word cloud object 
from wordcloud import WordCloud

wordcloud = WordCloud(colormap='prism', mask= mask_image, background_color='white')

wordcloud = wordcloud.generate(text)



