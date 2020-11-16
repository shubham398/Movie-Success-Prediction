import re 
import tweepy 
from textblob import TextBlob 
import numpy as np
from matplotlib import pyplot as plt

consumer_key = ''
consumer_secret = ''
access_token = ''
access_token_secret = ''

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth,wait_on_rate_limit=True)

countries=[
        {'name':'London','code':'51.51770,-0.11352,50km'}, 
        {'name':'Nepal','code':'27.72272,85.32002,50km'},
        {'name':'Singapore','code':'1.38982,103.98228,50km'},
        {'name':'Malaysia','code':'2.19744,102.249,50km'},
        {'name':'India','code':'28.64386,77.12373,50km'}]

sa_pos=[]
sa_neu=[]
sa_neg=[]
    
for c in countries:
    tweets=[]
    print(c['name'])
    for tweet in tweepy.Cursor(api.search,q="#Avengers:Endgame", lang="en", since="2019-03-08", until="2019-07-07", geocode=c['code']).items(300):
        tweets.append(tweet.text)
        #print(tweet.user)
        
    def clean_tweet(tweet):
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())
    
    def analize_sentiment(tweet):
        analysis = TextBlob(clean_tweet(tweet))
        if analysis.sentiment.polarity > 0:
            return 1
        elif analysis.sentiment.polarity == 0:
            return 0
        else:
            return -1
        
    
    sentimental_analysis = []
    sentimental_analysis = np.array([ analize_sentiment(tweet) for tweet in tweets ])
    
    pos_tweets = len([i for i in sentimental_analysis if i > 0]) 
    neu_tweets = len([i for i in sentimental_analysis if i == 0]) 
    neg_tweets = len([i for i in sentimental_analysis if i < 0]) 
    
    if len(tweets)>0:
        print("Percentage of positive tweets: {}%".format((pos_tweets)*100/len(sentimental_analysis)))
        print("Percentage of neutral tweets: {}%".format((neu_tweets)*100/len(sentimental_analysis)))
        print("Percentage de negative tweets: {}%".format((neg_tweets)*100/len(sentimental_analysis)))
        
        sa_pos.append((pos_tweets)*100/len(sentimental_analysis))
        sa_neu.append((neu_tweets)*100/len(sentimental_analysis))
        sa_neg.append((neg_tweets)*100/len(sentimental_analysis))
    else:
        print('No Data')
        sa_pos.append(0)
        sa_neu.append(0)
        sa_neg.append(0)
    
    
objects = ('London', 'Nepal', 'Singapore', 'Malaysia', 'India')
y_pos = np.arange(len(objects))
 
rec1=plt.bar(y_pos, sa_pos, align='center', alpha=0.5)
rec2=plt.bar(y_pos, sa_neg, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Usage')
plt.title('countries vs tweets')
plt.show()


