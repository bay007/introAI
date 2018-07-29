import tweepy
from textblob import TextBlob
import re
import os
consumer_key = os.getenv("CK")
consumer_secret = os.getenv("CKS")

acces_token = os.getenv("AT")
acces_token_secret = os.getenv("ATS")


auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(acces_token, acces_token_secret)

api = tweepy.API(auth)

public_tweets = api.search('Trump')


def clean(text):
    text = re.sub("\B(\#[a-zA-Z]+\b)(?!;)", "", text)
    text = re.sub("(^|[^@\w])@(\w{1,15})", "", text)
    text = re.sub(
        "(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9]\.[^\s]{2,})", "", text)
    text = re.sub("^RT:", "", text)
    text = text.replace("\n", "").replace("\t", "").replace("\r", "")

    return text.lstrip().rstrip()


with open("tweets.so", "w", encoding='utf-8') as f:
    for tweet in public_tweets:
        text = clean(tweet.text)
        analysis = TextBlob(text)
        print(analysis.sentiment)
        f.write("{} \t {} \t {} \r".format(
            text, analysis.sentiment.polarity, analysis.sentiment.subjectivity))
