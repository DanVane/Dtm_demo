# -*- coding: utf-8 -*-
from gensim import corpora
from gensim.models.wrappers.dtmmodel import DtmModel
import json
import re
import threading

data_path='E:\\NetBeans\\dianziyisuo\\python\\data\\jichengdianlu1.txt'
year_s = [2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017]


def read_from_file():
    t = open('data/stopwords.txt', 'rb')
    stoplist = []
    for line in t.readlines():
        line = line.strip().lower()
        stoplist.append(line)

    f=open(data_path,'rb')
    documents =[]
    i = 1
    for line in f.readlines():
        line=line.strip().lower()
        '''
        if i == 1:
            time_s = [int(w) for w in line.split(',')]
            i = i+1
            continue
            '''
        document=[]
        line = re.sub('\.', '', line)
        line = re.sub(r'[0-9]+', '', line)
        for w in line.split():
            if w not in stoplist:
                document.append(w)
        documents.append(document)
    corpus = DTMcorpus(documents)

    return corpus#, time_s

class DTMcorpus(corpora.textcorpus.TextCorpus):
    def get_texts(self):
        return self.input
    def __len__(self):
        return len(self.input)



def Model(time_s,ntopics):
    model = DtmModel('bin/dtm-win64.exe', corpus, time_slices=time_s, num_topics=ntopics,
                                     id2word=corpus.dictionary)
    return model

def getTopicIDs(ntopics):
    tid_list = [i for i in range(ntopics)]
    topicIDs = [
        {'topicID': tid_list},
    ]
    with open('results/jichengdianlu/all_topicIDs.json', 'w') as f:
        f.write(json.dumps(topicIDs))

def getTopic_Words(ntopics, time_s, nwords=10):
    topics = model.show_topics(num_topics=ntopics, times=len(time_s), num_words=nwords)
    for i in range(ntopics):
        year = []
        word = []
        for j in range(len(time_s)):
            year.append(year_s[j])
            oldS = topics[j*ntopics+i]
            #newS = re.sub('[^a-z+A-Z]', '', oldS)

            #newS = re.sub('[0-9]*[\.]?[0-9]*\*[0-9]*[\.]?[0-9]*]','',oldS)
            word.append(oldS.split('+'))
            #word.append(oldS)
        topic_words = [
            {'year': year, 'word':word},
        ]
        file_name= 'results/jichengdianlu/topic_%d_words_with_time.json'%i
        with open(file_name, 'w') as f:
            f.write(json.dumps(topic_words))
    return topic_words

def getTopic_Words_Effect(tid, nwords=10):
    year = []
    topic_effect_words = {}
    for t in range(len(time_s)):
        words = []
        probs = []
        year.append(year_s[t])
        topic = model.show_topic(topicid=tid , time = t, num_words=nwords)
        for c in range(len(topic)):
            words.append(topic[c][1])
            probs.append(topic[c][0])
        for i in range(len(words)):
            j = words[i]
            if j not in topic_effect_words:
                topic_effect_words[j] = {}
            topic_effect_words[j][t] = probs[i]

    t_series = {}
    for w, values in topic_effect_words.iteritems():
        series = []
        for t in range(len(time_s)):
            if t in values:
                series.append(values[t])
            else:  # No value for that time-period.
                series.append(0.)
        t_series[w] = series

    file_twe = [
            {'year': year, 'number': t_series},
        ]
    file_name= 'results/jichengdianlu/topic_%d_topic_words_effect.json'%tid
    with open(file_name, 'w') as f:
        f.write(json.dumps(file_twe))

# 读取语料，在data/foo-mul.dat
# 字典，在data/dictionary.dict

corpus = read_from_file()
time_s = [2000,  2000,  2000,  2000, 2000, 2000,   2000,   2000,  2000,    2000, 85]

ntopics = 5
model = Model(time_s,ntopics)
getTopicIDs(ntopics)
getTopic_Words(ntopics, time_s)
for i in range(5):
    getTopic_Words_Effect(i)

'''
if __name__=="__main__":
    while True:
        timer = threading.Timer()
        timer.start()
        timer.join()
'''




