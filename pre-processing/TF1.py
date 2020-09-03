import json
from collections import defaultdict

'''Create frequency dict vocabulary and apaply TF1 to remove words that only occur once'''


wordFrequency = defaultdict(int)

def remove_unique(tokensObj):
  updated_tokens = []
  for token in tokensObj:
    word = token['token']
    if wordFrequency[word] > 1:
      updated_tokens.append(token)

  cleaned_text = [token['token'] for token in updated_tokens]
  cleaned_text = " ".join(cleaned_text)

  return updated_tokens, cleaned_text


def TF1(data):
  # create word frequency dictionary
  for el in data:
    for token in el['tokens']:
      word = token['token']
      wordFrequency[word] += 1
  
  # TF1: 
  # return non-unique words
  vocabulari_after_tf1 = [value for key,value in wordFrequency.items() if value > 1 ]
  print('vocabulari size: ', len(wordFrequency))
  print('vocabulari size after TF1: ', len(vocabulari_after_tf1))

  # remove unique words from pre-processed data
  for el in data:
    cleaned_tokens, cleaned_text = remove_unique(el['tokens'])
    el['tokens'] = cleaned_tokens
    el['cleaned_text'] = cleaned_text

  return data