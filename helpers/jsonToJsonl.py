'''
  This script is made to convert json files from pre-processing to jsonl files,
  which is the format that the data passed to torchtext needs to have
'''

import sys
import json

def jsonToJsonl(inputFile, outputFile):
  with open(inputFile, 'r') as json_file:
      data = json.load(json_file)

  with open(outputFile, 'w') as trainFile:
      for entry in data['data']:
          json.dump(entry, trainFile)
          trainFile.write('\n')

if __name__ == "__main__":
  inputFile = sys.argv[1]
  print(inputFile)
  dataset = inputFile.split('.')[0]
  outputFile = f'{dataset}.jsonl'
  jsonToJsonl(inputFile, outputFile)