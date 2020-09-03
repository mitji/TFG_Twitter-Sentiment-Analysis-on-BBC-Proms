import codecs
import unicodedata
import emoji

train_2013 = './../../data/training/raw data/semeval/2017_English_final/GOLD/Subtask_A/twitter-2013train-A.txt';
dev_2013 = './../../data/training/raw data/semeval/2017_English_final/GOLD/Subtask_A/twitter-2013dev-A.txt';
test_2013 = './../../data/training/raw data/semeval/2017_English_final/GOLD/Subtask_A/twitter-2013test-A.txt';
test_2013_2 = './../../data/training/raw data/semeval/2017_English_final/GOLD/Subtask_A/sms-2013test-A.txt';
test_2014 = './../../data/training/raw data/semeval/2017_English_final/GOLD/Subtask_A/twitter-2014test-A.txt';
test_2014_2 = './../../data/training/raw data/semeval/2017_English_final/GOLD/Subtask_A/livejournal-2014test-A.txt';
train_2015 = './../../data/training/raw data/semeval/2017_English_final/GOLD/Subtask_A/twitter-2015train-A.txt';
test_2015 = './../../data/training/raw data/semeval/2017_English_final/GOLD/Subtask_A/twitter-2015test-A.txt';
train_2016 = './../../data/training/raw data/semeval/2017_English_final/GOLD/Subtask_A/twitter-2016train-A.txt';
dev_2016 = './../../data/training/raw data/semeval/2017_English_final/GOLD/Subtask_A/twitter-2016dev-A.txt';
devTest_2016 = './../../data/training/raw data/semeval/2017_English_final/GOLD/Subtask_A/twitter-2016devtest-A.txt';
test_2016 = './../../data/training/raw data/semeval/2017_English_final/GOLD/Subtask_A/twitter-2017-task4-dev.subtask-A.english.INPUT.txt';

raw_data_files = [train_2013, dev_2013, test_2013, test_2013_2, test_2014, test_2014_2, train_2015, test_2015, train_2016, dev_2016, devTest_2016, test_2016]
raw_test_file = './../../data/test/SemEval2017-task4-test.subtask-A.english.txt'

count_tweets = 0
training_file = 'raw_data_training.txt'
test_file = 'raw_data_test.txt'

# with open(training_file, 'w', encoding="utf-8") as training:
#   for index, file in enumerate(raw_data_files):
#     f = open(file, encoding = "utf-8")
#     print(f'--- opened {file}')
#     for line in f:
#       count_tweets += 1
#       # change multiple white spaces to single white spaces to separate correctly id from sentiment
#       formated_line = ' '.join(line.split())
#       # separate id - sentiment - text
#       if index == 3 or index == 5:
#         splited_line = formated_line.split(' ', 3)  # split by single space only twice = array of [id, sentiment, text]
#         # take only sentiment and text
#         sentiment = splited_line[2]
#         # clean text
#         text = splited_line[3]
#       else:
#         splited_line = formated_line.split(' ', 2)  # split by single space only twice = array of [id, sentiment, text]
#         # take only sentiment and text
#         sentiment = splited_line[1]
#         # clean text
#         text = splited_line[2]

#       text = emoji.demojize(text)
#       training.write(f'{sentiment} {text}\r\n')
#     print(f'processed {file}')
#     f.close()

# with open(test_file, 'w') as test:
#     f = open(raw_test_file, encoding = "utf-8")
#     for line in f:
#       count_tweets += 1
#       # change multiple white spaces to single white spaces to separate correctly id from sentiment
#       formated_line = ' '.join(line.split())
#       # separate id - sentiment - text
#       splited_line = formated_line.split(' ', 2)  # split by single space only one = array of [sentiment, text]
#       # take only sentiment and text
#       sentiment = splited_line[1]
#       # clean text
#       text = splited_line[2]
#       text = emoji.demojize(text)
#       test.write(f'{sentiment} {text}\r\n')
#     print(f'processed')
#     f.close()

# BBC PROMS
bbcproms_annotated = './../../data/BBCProms/annotated.txt';
bbcproms_output = 'bbcProms.txt'
with open(bbcproms_output, 'w') as test:
    f = open(bbcproms_annotated, encoding = "utf-8")
    for line in f:
      count_tweets += 1
      # change multiple white spaces to single white spaces to separate correctly id from sentiment
      formated_line = ' '.join(line.split())
      # separate id - sentiment - text
      splited_line = formated_line.split(' ', 1)  # split by single space only one = array of [sentiment, text]
      # take only sentiment and text
      sentiment = splited_line[0]
      # clean text
      text = splited_line[1]
      text = emoji.demojize(text)
      test.write(f'{sentiment} {text}\r\n')
    print(f'processed')
    f.close()

print(count_tweets)