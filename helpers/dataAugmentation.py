'''
  Script adapted from:
  https://github.com/kothiyayogesh/medium-article-code/blob/master/How%20I%20dealt%20with%20Imbalanced%20text%20dataset/data_augmentation_using_language_translation.ipynb
  But using googletrans instead of TextBlob translator: https://py-googletrans.readthedocs.io/en/latest/ 
'''

import nltk
from googletrans import Translator
from googletrans import LANGUAGES
import random
import numpy as np
import json
import time
from nltk.stem import WordNetLemmatizer
from posTagger import runtagger_parse

input_file = './reduced.json'
output_file = "reduced_augmented.json"

# Translator
sr = random.SystemRandom()
languages = [lang for lang in LANGUAGES]
translator = Translator()
def generateSynthTweets(message, aug_range=3, languages=languages):
  augmented_messages = []

  for j in range(0,aug_range) :
    language = sr.choice(languages)
    try:
      translated = translator.translate(message, src="en", dest=language).text
      new_text = translator.translate(translated, src=language, dest="en").text
      augmented_messages.append(str(new_text))
    except Exception as e:
      print('Error in generateSynthTweets:', e)

  return augmented_messages