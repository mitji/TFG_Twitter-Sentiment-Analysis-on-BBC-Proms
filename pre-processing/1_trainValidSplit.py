import json
from sklearn.model_selection import train_test_split

from TF1 import TF1

raw_training = open("./raw_data_training.txt")

# split train and validation before pre-processing, generate .txt files
train_file = 'train.txt'
valid_file = 'validation.txt'
tweets = []
for line in raw_training:
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

  # create little tweet object and save in tweets array. that way is easier to make the split
  tweet['sentiment'] = sentiment
  tweet['raw_text'] = text
  tweets.append(tweet)

raw_training.close()

# split data
train_data, valid_data = train_test_split(tweets, test_size=0.15, random_state=25)

with open(train_file, 'w') as train:
  for el in train_data:
    sentiment = el['sentiment']
    text = el ['raw_text']
    train.write(f'{sentiment} {text}\r\n')
  print(f'train set created')

with open(valid_file, 'w') as valid:
  for el in valid_data:
    sentiment = el['sentiment']
    text = el ['raw_text']
    valid.write(f'{sentiment} {text}\r\n')
  print(f'validation set created')

print(len(train_data))
print(len(valid_data))

# with open(input, 'r') as json_file:
#     data = json.load(json_file)

# dataset = data['data']

# # Apply TF1 on training data
# train_data_tf1 = TF1(train_data)
# with open('train.jsonl', 'w') as trainFile:
#     for entry in train_data:
#         json.dump(entry, trainFile)
#         trainFile.write('\n')

# with open('valid.jsonl', 'w') as validData:
#     for entry in valid_data:
#         json.dump(entry, validData)
#         validData.write('\n')

# with open('test.jsonl', 'w') as testFile:
#     for entry in dataset:
#         json.dump(entry, testFile)
#         testFile.write('\n')

# with open('bbc-proms.jsonl', 'w') as testBBCProms:
#     for entry in dataset:
#         json.dump(entry, testBBCProms)
#         testBBCProms.write('\n')