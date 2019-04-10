import sys

from tqdm import tqdm
from twitterscraper import query_tweets
from twitterscraper.ts_logger import logger

logger.propagate = False
logger.disabled = True

with open('common_' + sys.argv[1] + '.txt', encoding='ISO-8859-1') as f:
    word_list = f.readlines()

i = 0
pbar = tqdm()
for word in reversed(word_list):
    word = word.strip()
    if len(word) <= 2: continue

    with open("result_" + sys.argv[1] + '.txt', "a") as f:
        for tweet in query_tweets(word, 10000, lang=sys.argv[1]):
            f.write(tweet.text + "\n")
            pbar.update(1)
