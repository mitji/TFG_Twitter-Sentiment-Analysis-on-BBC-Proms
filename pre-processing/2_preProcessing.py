""" 
This file  prepares the data for the classifiers.
STEPS: 
  1) Text pre-processing
  2) POS Tagger & tokenizer (see posTagger.py)
  3) Test post-processing

RESTRICTIONS:
  - input file: each line must have the following format sentiment<SPACE>text
  - output file: json file with all the needed information of the tweet
"""

import json
import re
import nltk
from nltk.stem import WordNetLemmatizer
from ekphrasis.classes.segmenter import Segmenter
from pycontractions import Contractions
import time
import codecs
import unidecode
import emoji
import random
import numpy as np

from posTagger import runtagger_parse
from dataAugmentation import generateSynthTweets
from TF1 import TF1

result = dict()
result['data'] = []

inputFile = open("./train.txt")
'''
  Change output dataset to apply different processings to the data
  Options:
    - training
    - validation
    - test
    - training-binary
    - validation-binary
    - test-binary
    - training-augmented
'''
dataset = "training"
output_file = "bbcProms.json"

EMOTICONS = {
    ":)": "grinning face",
    "(:": "grinning face",
    ":-)": "grinning face",
    ":-]": "grinning face",
    ":-3": "grinning face",
    ":->": "grinning face",
    ":-}": "grinning face",
    "8-)": "grinning face",
    ":]": "grinning face",
    ":3": "grinning face",
    ":>": "grinning face",
    "8)": "grinning face",
    ":}": "grinning face",
    "=]": "grinning face",
    "=)": "grinning face",
    ":o)": "grinning face",
    ":c)": "grinning face",
    ":^)": "grinning face",
    ":D": "beam face with smile eyes",
    ":-D": "beam face with smile eyese",
    ":'-)": "beam face with smile eyes",
    ":')": "beam face with smile eyes",
    "xd": "grinning face",
    "XD": "grinning face",
    "xD": "grinning face",
    "Xd": "grinning face",
    "8D": "grinning face",
    "=D": "beam face with smile eyes",
    "=3": "beam face with smile eyes",
    ":'-)": "beam face with smile eyes",
    ":')": "beam face with smile eyes",
    ":-))": "beam face with smile eyes",
    ":-(": "frown face",
    ":‑c": "frown face",
    ":c	": "frown face",
    ":‑<": "frown face",
    ":<": "frown face",
    ":‑[": "frown face",
    ":[	": "frown face",
    ":-||": "frown face",
    ">:[	": "frown face",
    ":{	": "frown face",
    ":@	": "frown face",
    ">:(	": "frown face",
    ";(": "frown face",
    ":'‑(": "very frown face",
    ":'(": "very frown face",
    ":‑c": "very frown face",
    ":c": "very frown face",
    "(>_<)": "very frown face",
    "(>_<)>": "very frown face",
    "(T_T)": "crying face",
    "T_T": "crying face",
    "D‑':": "frown face",
    "D:<": "frown face",
    "D:": "frown face",
    "D8": "frown face",
    "D;": "frown face",
    "D=": "frown face",
    "DX": "frown face",
    "O_O": "frown face",
    "o-o": "frown face",
    "O_o": "frown face",
    "o_O": "frown face",
    "o_o": "frown face",
    "O-O": "frown face",
    ":‑O": "frown face",
    ":O": "frown face",
    ":‑o": "frown face",
    ":o": "frown face",
    ":-0": "frown face",	
    "8‑0": "frown face",	
    ">:O": "frown face",
    ":-*": "kiss",
    ":*":"kiss",
    ":×":"kiss",
    ";‑)": "smirking face",
    ";)": "smirking face",
    "*-)": "smirking face",
    "*)	": "smirking face",
    ";‑]": "smirking face",
    ";]": "smirking face",
    ";^)": "smirking face",
    ":‑,": "smirking face",
    ";D": "smirking face",
    ":‑P": "smirking face",
    ":P": "smirking face",
    "X‑P": "smirking face",
    "XP": "smirking face",
    "x‑p": "smirking face",
    "xp": "smirking face",
    ":‑p": "smirking face",
    ":p": "smirking face",
    ":‑Þ": "smirking face",
    ":Þ": "smirking face",
    ":‑þ": "smirking face",
    ":þ	": "smirking face",
    ":‑b": "smirking face",
    ":b	": "smirking face",
    "d:	": "smirking face",
    "=p": "smirking face",
    ">:P": "smirking face",
    ">.<": "frown face",
    ":‑/": "frown face",
    ":/": "frown face",
    ":‑.": "frown face",
    ">:/": "frown face",
    "=/": "frown face",
    ":L": "frown face",
    "=L": "frown face",
    ":S": "frown face",
    ":‑|": "neutral",
    ":|": "neutral",
    ":$": "flushed face",
    "://)": "flushed face",
    "://3": "flushed face",
    "|;‑)": "smiling face with sunglasses",	
    "|‑O": "sleepy face",
    ":‑J": "beam face with smile eyes",
    "%‑)": "frown face",
    "%)": "frown face",
    "',:-|": "face with raised eyebrow",
    "',:-l": "face with raised eyebrow",
    "(._.)": "face with raised eyebrow",
    "(-_-)!!": "face with raised eyebrow",
    "._.": "face with raised eyebrow",
    "-_-": "face with raised eyebrow",
    "-.-": "face with raised eyebrow",
    "(-.-)": "face with raised eyebrow",
    "(-_-)": "face with raised eyebrow",
    "(--)": "face with raised eyebrow",
    ">_>": "face with raised eyebrow",
    "<_>": "face with raised eyebrow",
    "</3": "broken heart",
    "<3": "red heart",
    "*_*": "heart",
    "(=_=)": "tired face",
    "(・・?": "frown face"
  }
CONTRACTION_MAP = {
  "ain't": "is not",
  "aren't": "are not",
  "can't": "cannot",
  "can't've": "cannot have",
  "'cause": "because",
  "could've": "could have",
  "couldn't": "could not",
  "couldn't've": "could not have",
  "didn't": "did not",
  "doesn't": "does not",
  "don't": "do not",
  "hadn't": "had not",
  "hadn't've": "had not have",
  "hasn't": "has not",
  "haven't": "have not",
  "he'd": "he would",
  "he'd've": "he would have",
  "he'll": "he will",
  "he'll've": "he he will have",
  "he's": "he is",
  "how'd": "how did",
  "how'd'y": "how do you",
  "how'll": "how will",
  "how's": "how is",
  "I'd": "I would",
  "I'd've": "I would have",
  "I'll": "I will",
  "I'll've": "I will have",
  "I'm": "I am",
  "I've": "I have",
  "i'd": "i would",
  "i'd've": "i would have",
  "i'll": "i will",
  "i'll've": "i will have",
  "i'm": "i am",
  "i've": "i have",
  "isn't": "is not",
  "it'd": "it would",
  "it'd've": "it would have",
  "it'll": "it will",
  "it'll've": "it will have",
  "it's": "it is",
  "let's": "let us",
  "ma'am": "madam",
  "mayn't": "may not",
  "might've": "might have",
  "mightn't": "might not",
  "mightn't've": "might not have",
  "must've": "must have",
  "mustn't": "must not",
  "mustn't've": "must not have",
  "needn't": "need not",
  "needn't've": "need not have",
  "o'clock": "of the clock",
  "oughtn't": "ought not",
  "oughtn't've": "ought not have",
  "shan't": "shall not",
  "sha'n't": "shall not",
  "shan't've": "shall not have",
  "she'd": "she would",
  "she'd've": "she would have",
  "she'll": "she will",
  "she'll've": "she will have",
  "she's": "she is",
  "should've": "should have",
  "shouldn't": "should not",
  "shouldn't've": "should not have",
  "so've": "so have",
  "so's": "so as",
  "that'd": "that would",
  "that'd've": "that would have",
  "that's": "that is",
  "there'd": "there would",
  "there'd've": "there would have",
  "there's": "there is",
  "they'd": "they would",
  "they'd've": "they would have",
  "they'll": "they will",
  "they'll've": "they will have",
  "they're": "they are",
  "they've": "they have",
  "to've": "to have",
  "wasn't": "was not",
  "we'd": "we would",
  "we'd've": "we would have",
  "we'll": "we will",
  "we'll've": "we will have",
  "we're": "we are",
  "we've": "we have",
  "weren't": "were not",
  "what'll": "what will",
  "what'll've": "what will have",
  "what're": "what are",
  "what's": "what is",
  "what've": "what have",
  "when's": "when is",
  "when've": "when have",
  "where'd": "where did",
  "where's": "where is",
  "where've": "where have",
  "who'll": "who will",
  "who'll've": "who will have",
  "who's": "who is",
  "who've": "who have",
  "why's": "why is",
  "why've": "why have",
  "will've": "will have",
  "won't": "will not",
  "won't've": "will not have",
  "would've": "would have",
  "wouldn't": "would not",
  "wouldn't've": "would not have",
  "y'all": "you all",
  "y'all'd": "you all would",
  "y'all'd've": "you all would have",
  "y'all're": "you all are",
  "y'all've": "you all have",
  "you'd": "you would",
  "you'd've": "you would have",
  "you'll": "you will",
  "you'll've": "you will have",
  "you're": "you are",
  "you've": "you have"
}
lemmatizer = WordNetLemmatizer() # you need to download the wordent package --> run nltk.download('wordnet') in the command line
seg_tw = Segmenter(corpus="twitter")
# Load semantic vector model from the gensim.downloader api
cont = Contractions(api_key="glove-twitter-100")
# prevents loading on first expand_texts call
cont.load_models()
# custom_stopwords = ['the', 'be', 'to', 'in', 'and', 'a', 'i', 'with', 'on', 'of', 'a', 'go', 'for', 'tomorrow', 'that',]


def lemmatizeWord(word, pos):
  if pos == 'NOUN': return lemmatizer.lemmatize(word, pos="n")
  if pos == 'VERB': return lemmatizer.lemmatize(word, pos="v")
  if pos == 'ADJECTIVE': return lemmatizer.lemmatize(word, pos="a")
  if pos == 'ADVERB': return lemmatizer.lemmatize(word, pos="r")

def fixWordLength(word):
  pattern = re.compile(r"(.)\1{2,}")
  return pattern.sub(r"\1\1", str(word))

def initialCleaning(text): 
  # 1.1. Convert emoticon to sentiment text
  # text = emoji.demojize(text) # no funciona aqui no se pq
  text = text.replace('\\', "")
  words = text.split()
  text_without_emoticons = [EMOTICONS[word] if word in EMOTICONS else word for word in words]
  text = " ".join(text_without_emoticons)
  # 1.2. Expand contractions before tokenization
  expanded_text = list(cont.expand_texts([text]))[0]
  # Extend remainign ones with custom contractions dictionary
  text_without_contractions = [CONTRACTION_MAP[word] if word in CONTRACTION_MAP else word for word in expanded_text.split()]
  cleaned_text_1 = " ".join(text_without_contractions)
  return cleaned_text_1

def finalCleaning(tokensObj, text):
  copyTokensObj = tokensObj
  for index, token in enumerate(copyTokensObj):
    # 2. Segment hashtags and convert generated words to tokens
    if token['tag'] == 'HASHTAG':
      segmented = seg_tw.segment(token['token'][1:]) # ignore '#' from the beginning of the token
      # generate tokens
      words = segmented.split()
      count = 1
      for word in words:
        newToken = { 'token': word, 'tag': 'HASHTAG'}
        tokensObj.insert(index+count, newToken)
      # at the end, remove original hashtag
      tokensObj.pop(index)

    # split texts converted by emoji and generate new emoji token (only for test and BBCProms, avoid in training, no emojis in dataset)
    if dataset == 'test':
      if "_" in token['token'] and token['tag'] not in ['HASHTAG', 'AT-MENTION', 'URL/EMAIL', 'DISCOURSE MARKER']:
        emoji_split = token['token'].split("_")
        emoji = " ".join(emoji_split)
        newToken = { 'token': emoji, 'tag': 'emoji'}
        tokensObj.insert(index+1, newToken)
        tokensObj.pop(index)

  filtered_tokens = []
  for index, token in enumerate(tokensObj):
    # 3. generate EMPHASIS tokens
    # 3.1 convert ! to emphasis tokens
    if token['tag'] == 'PUNCTUATION' and '!' in token['token']:
      token['tag'] = 'EMPHASIS'
      if len(token['token']) > 1:
        for char in token['token']:
          newToken = { 'token': char, 'tag': 'EMPHASIS'}
          filtered_tokens.append(newToken)
        continue

    # 4. Remove unwanted tokens
    if token['tag'] not in ['PUNCTUATION','AT-MENTION', 'URL/EMAIL', 'NUMERAL', 'DISCOURSE MARKER', 'OTHERS']:
      # 3.2 convert CAPITALIZED words to emphasis tokens
      if token['token'].isupper() and token['tag'] not in ['EMPHASIS'] and token['token'] is not 'I':
        # avoid taking first word of string as an emphasis word (ex: 'A great day', 'A' is not taken as an emphasis word)
        if index != 0 or len(token['token'])>1:
          token['tag'] = 'EMPHASIS'
      
      # 5. Convert words with same consecutive vowels to emphasis tokens (ex: soooo --> EMPHASIS TOKEN)
      word = token['token']
      vowels = 'aeiou'
      for i in range(len(word)-2):
        if word[i] in vowels and word[i] == word[i+1] and word[i] == word[i+2]:
          token['tag'] = 'EMPHASIS'
          break
      
      # 6. Vocabulary reduction
      # 6.1 Spelling correction (i don't want to correct proper nouns and hashtags, since hashtags are usually not mis-spelled
      if token['tag'] not in ['PROPER NOUN', 'HASHTAG']:
        # 6.1.1: fix word length
        token['token'] = fixWordLength(token['token'])
        # 6.1.2 spell correction
        # token['token'] = str(TextBlob(token['token']).correct())
      
      # 6.2 apply lemmatization to get the root of words and reduce vocabulary size
      if token['tag'] in ['NOUN', 'VERB', 'ADJECTIVE', 'ADVERB']:
        token['token'] = lemmatizeWord(token['token'], token['tag'])

      # 7. Substitute emoticons detected by the tokenizer with the emoticon sentiment. 
      # If is not in my emoticons list, create empty token
      if token['tag'] == 'EMOTICON':
        emoticon = token['token']
        emoticonToSentiment = EMOTICONS[emoticon] if emoticon in EMOTICONS else ''
        token['token'] = emoticonToSentiment

      # 8. Finally lowercase if token is not an emphasis or emoticon token
      # if token['tag'] not in ['EMPHASIS', 'EMOTICON']: # removed this to reduce vocabulary size
      token['token'] = token['token'].lower()
      
      # 9. Remove custom stopwords list
      #if token['token'] not in custom_stopwords : filtered_tokens.append(token)
      filtered_tokens.append(token)
  
  non_empty_tokens = [token for token in filtered_tokens if token['token'] is not ""]
  cleaned_text = [token['token'] for token in non_empty_tokens]
  cleaned_text = " ".join(cleaned_text)
  return non_empty_tokens, cleaned_text

if __name__ == "__main__":
  tweets = []
  start = time.time()

  # Step 1 Data Cleaning
  print('Starting cleaning step 1!\n')
  for line in inputFile:
    tweet = dict()
    # change multiple white spaces to single white spaces to separate correctly id from sentiment
    formated_line = ' '.join(line.split())
    # separate id - sentiment - text
    splited_line = formated_line.split(' ', 1)  # split by single space only once = array of [sentiment, text]
    # take only sentiment and text
    sentiment = splited_line[0]
    # clean text
    text = splited_line[1]
    text = text.strip()                     # remove whitespaces from beginning and end
    text = text.strip('"')                  # remove character from beginning and end
    text = text.replace('"', "'")
    # try:
    #     decoded = unidecode.unidecode(codecs.decode(text, 'unicode_escape'))
    # except:
    #     decoded = unidecode.unidecode(text)

    # BINARY CLASSIFICATION
    #if sentiment != 'neutral':
    tweet['sentiment'] = sentiment
    tweet['raw_text'] = text
    cleaned = initialCleaning(text)
    tweet['cleaned_text'] = cleaned
    result['data'].append(tweet)
    tweets.append(cleaned)
    
  print('Data cleaning step 1 finished!\n')

  ###################
  if dataset == 'training-augmented':
    # START - Data augmentation
    print('START - Generating new tweets...')

    # declare & initialize variables
    neg_tweets = [el for el in result['data'] if el['sentiment'] == 'negative']
    synthetic_tweets = []
    count = 0
    num_of_new_tweets = 5250  # fit pos samples = 10449 new samples, fit neutral samples = 13325

    # random indexes
    selected_indexes = [] # check list to store generated indexes so they are not repeated
    for tweet in neg_tweets:
      # index = random.randint(0, len(neg_tweets) - 1)
      # if index not in selected_indexes:
        # Create synthetic tweets
        # - generate synthetic tweet samples
      new_tweets = generateSynthTweets(tweet['cleaned_text'], 2)
      # - append synthetic tweets to complete list of synth tweets
      # way to do the spread operator in python
      if len(new_tweets) > 0:
        # update loop variables
        # selected_indexes.append(index)
        synthetic_tweets = [*synthetic_tweets, *new_tweets]
        count += 1
        print('translated!', count)
      else:
        continue

    # take number of interest of random tweets as new generated data
    selected_tweets = np.take(synthetic_tweets, np.random.permutation(len(synthetic_tweets))[:num_of_new_tweets])
    for index, el in enumerate(selected_tweets):
      new_tweet = dict()
      new_tweet['sentiment'] = 'negative'
      new_tweet['raw_text'] = el
      new_tweet['cleaned_text'] = initialCleaning(el)
      new_tweet['augmented'] = True
      result['data'].append(new_tweet)
      tweets.append(initialCleaning(el))

    print('FINISH - Generating new tweets...\n')

  # END - Data augmentation
  ###################

  # Step 2 Tokenization: tokenizer + PoS Tagger
  print('Starting tokenizer...\n')
  tokenization_results = runtagger_parse(tweets)
  end_tokenizer = time.time()
  print('Tokenization and pos tagger finished in ', end_tokenizer - start, 's')
  # Step 3 Text pre-processing: main part
  print('\nStarting cleaning step 3!\n')
  for index, el in enumerate(result['data']):
    el['tokens'] = tokenization_results[index]
    cleaned_tokens, cleaned_text = finalCleaning(el['tokens'], el['cleaned_text'])
    el['tokens'] = cleaned_tokens
    el['cleaned_text'] = cleaned_text
  
  # Apply TF1 on training dataset
  if dataset == 'training' or dataset == 'training-augmented':
    print('Starting TF1')
    result['data'] = TF1(result['data'])
    print('Finish TF1')

  # create input-tweets-data.json with all the data
  with open(output_file, 'w') as json_file:
    json.dump(result, json_file, indent=3, ensure_ascii=False)
  
  end = time.time()
  # See number of tweets
  negative = [el for el in result['data'] if el['sentiment'] == 'negative']
  neutral = [el for el in result['data'] if el['sentiment'] == 'neutral']
  positive = [el for el in result['data'] if el['sentiment'] == 'positive']
  print(f'negative: {len(negative)} - neutral: {len(neutral)} - positive: {len(positive)}')

  print('Total elapsed time: ', end - start, 's')
