#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Python wrapper for runTagger.sh script for CMU's Tweet Tokeniser and Part of Speech tagger: http://www.ark.cs.cmu.edu/TweetNLP/

Script adapted from https://github.com/ianozsvald/ark-tweet-nlp-python to adapt it into my pre-processing pipeline from '2_preProcessing.py", 
and make it callable from other files
Usage:
tokens = runtagger_parse(list_of_tweets)
where list_of_tweets is a list of strings
results will contain a list dicts/objects (one per tweet) with two propertioes, one for the token itself and one for the POS tag
"""
import subprocess
import shlex

# The only relavent source I've found is here:
# http://m1ked.com/post/12304626776/pos-tagger-for-twitter-successfully-implemented-in
# which is a very simple implementation, my implementation is a bit more
# useful (but not much).

# NOTE this command is directly lifted from runTagger.sh
RUN_TAGGER_CMD = "java -XX:ParallelGCThreads=2 -Xmx500m -jar ./../ark-tweet-nlp-python/ark-tweet-nlp-0.3.2.jar"

"""Function to rename the tag type to be more descriptive"""
def retag(tag):
    tagDictionary = {
        'N': 'NOUN',
        'O': 'PRONOUN',
        '^': 'PROPER NOUN',
        'S': 'NOMINAL + possesive',
        'Z': 'PROPER NOUN + POSSESSIVE',
        'V': 'VERB',
        'L': 'NOMINAL + VB',
        'M': 'PROPER NOUN + VB',
        'A': 'ADJECTIVE',
        'R': 'ADVERB',
        '!': 'INTERJECTION',
        'D': 'DETERMINER',
        'P': 'PRE/POST-POSITION',
        '&': 'CONJUNCTION',
        'T': 'VERB PARTICLE',
        'X': 'PREDETERMINER',
        'Y': 'PREDTERMINER+VERB',
        '#': 'HASHTAG',
        '@': 'AT-MENTION',
        '~': 'DISCOURSE MARKER',
        'U': 'URL/EMAIL',
        'E': 'EMOTICON',
        '$': 'NUMERAL',
        ',': 'PUNCTUATION',
        'G': 'OTHERS',
    }

    return tagDictionary.get(tag, tag)

def _split_results(rows):
    """Parse the tab-delimited returned lines, modified from: https://github.com/brendano/ark-tweet-nlp/blob/master/scripts/show.py"""
    for line in rows:
        line = line.strip()  # remove '\n'
        if len(line) > 0:
            if line.count('\t') == 2:
                parts = line.split('\t')
                tokens = parts[0]
                tags = retag(parts[1])
                confidence = float(parts[2])
                d = dict();
                d['token'] = tokens
                d['tag'] = tags
                yield d
                #yield tokens, tags, confidence


def _call_runtagger(tweets, run_tagger_cmd=RUN_TAGGER_CMD):
    """Call runTagger.sh using a named input file"""

    # remove carriage returns as they are tweet separators for the stdin
    # interface
    tweets_cleaned = [tw.replace('\n', ' ') for tw in tweets]
    message = "\n".join(tweets_cleaned)

    # force UTF-8 encoding (from internal unicode type) to avoid .communicate encoding error as per:
    # http://stackoverflow.com/questions/3040101/python-encoding-for-pipe-communicate
    message = message.encode('utf-8')

    # build a list of args
    args = shlex.split(run_tagger_cmd)
    args.append('--output-format')
    args.append('conll')
    po = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # old call - made a direct call to runTagger.sh (not Windows friendly)
    #po = subprocess.Popen([run_tagger_cmd, '--output-format', 'conll'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    result = po.communicate(message)
    # expect a tuple of 2 items like:
    # ('hello\t!\t0.9858\nthere\tR\t0.4168\n\n',
    # 'Listening on stdin for input.  (-h for help)\nDetected text input format\nTokenized and tagged 1 tweets (2 tokens) in 7.5 seconds: 0.1 tweets/sec, 0.3 tokens/sec\n')

    pos_result = result[0].decode('utf-8').strip('\n\n')  # get first line, remove final double carriage return
    pos_result = pos_result.split('\n\n')  # split messages by double carriage returns
    pos_results = [pr.split('\n') for pr in pos_result]  # split parts of message by each carriage return
    return pos_results


def runtagger_parse(tweets, run_tagger_cmd=RUN_TAGGER_CMD):
    """Call runTagger.sh on a list of tweets, parse the result, return lists of tuples of (term, type, confidence)"""    
    pos_raw_results = _call_runtagger(tweets, run_tagger_cmd)
    pos_result = []
    for pos_raw_result in pos_raw_results:
        pos_result.append([x for x in _split_results(pos_raw_result)])
    return pos_result
