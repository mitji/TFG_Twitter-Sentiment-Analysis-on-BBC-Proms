import xml.etree.ElementTree as ET
import json

tree = ET.parse('./../../data/ML-SentiCon/senticon.en.xml')
root = tree.getroot()
pos = {}
neg = {}

for el in root:
  for subel in el:
    if subel.tag == 'positive':
      for lemma in subel:
        word = lemma.text.strip()
        polarity = lemma.attrib['pol']
        if word in pos:
          if polarity > pos[word]:
            pos[word] = polarity
        else:
          pos[word] = polarity
    else:
      for lemma in subel:
        word = lemma.text.strip()
        polarity = lemma.attrib['pol']
        if word in neg:
          if polarity > neg[word]:
            neg[word] = polarity
        else:
          neg[word] = polarity

lexicon = {}
lexicon['positive'] = pos
lexicon['negative'] = neg

with open('./../../data/senticon-en.json', 'w') as json_file:
    json.dump(lexicon, json_file, indent=3, ensure_ascii=False)
