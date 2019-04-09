from twitterscraper import query_tweets

with open('common_en.txt', encoding='utf-') as f:
    en_list = f.readlines()

with open('common_es.txt', encoding='ISO-8859-1') as f:
    es_list = f.readlines()

for en in en_list[:200]:

    en = en.strip()
    if len(en) <= 2: continue

    for es in es_list[:200]:
        es = es.strip()
        if len(es) <= 2: continue

        with open("result.txt", "a") as f:
            for tweet in query_tweets(en + " AND " + es, 10000, poolsize=40):
                f.write(tweet.id + "\t" + tweet.text + "\n")

        print(en + " AND " + es + "\n ")
