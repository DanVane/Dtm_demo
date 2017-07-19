#coding=utf-8
from gensim.models import Word2Vec
import json
import re
import string

def load_data():
    f=open("E:\\NetBeans\\dianziyisuo\\python\\data\\jichengdianlu1.txt",'rb')
    lines = f.readlines()
    # lines = re.sub(string.punctuation,"",lines)
    # lines = re.sub(r'[0-9]+', ' ', lines)
    all_lines=[]
    for line in lines:
        line = line.strip().lower()
        line = re.sub(r'[0-9]+','',line)

        line = line.replace(string.punctuation,"")
        all_lines.append(line.split())




    model = Word2Vec(all_lines, size=100, window=5, min_count=0, workers=4)
    fname="w2v_models/w2v.model"
    model.save(fname)


def cal_sim():

    """
    do something ...
    :return:
    """
    fname = "w2v_models/w2v.model"
    model = Word2Vec.load(fname)
    for i in range(20):
        filename = "results/jichengdianlu/topic_"+str(i)+"_words_with_time.json"
        f=open(filename,'rb')
        data= json.load(f)[0]
        data = data['word']
        all_data=[]
        for y_data in data:
            n_data =[word.split('*')[1] for word in y_data]
            all_data.append(n_data)
        first_words = []
        second_words = []
        for j in range(9):
            result = dict([])
            for w1 in all_data[j]:
                for w2 in all_data[j+1]:
                    if w1.strip()==w2.strip():
                        sim=0
                    else:
                        try:
                            sim = model.wv.similarity(w2.strip(), w1.strip())
                        except Exception, e:
                            sim=0
                            pass
                    w = w1.strip()+':'+w2.strip()
                    result[w]=sim
            ll = sorted(result.iteritems(), key=lambda d: d[1])[::-1]
            first_words.append(ll[0])
            second_words.append(ll[1])
        sim_words = [
            {'fist_words':first_words, 'second_words':second_words},
        ]
        file_name = 'results/jichengdianlu/topic_%d_year_words_similarity.json' % i
        with open(file_name, 'w') as f:
            f.write(json.dumps(sim_words))

load_data()
cal_sim()

