from gensim import models,corpora
from six import iteritems

class MyCorpus(object):
    def __iter__(self):
        for line in open('mycorpus.txt'):
            # assume there's one document per line, tokens separated by whitespace
            yield dictionary.doc2bow(line.lower().split())

corpus_memory_friendly = MyCorpus()

stoplist = set('for a of the and to in'.split())

 # collect statistics about all tokens
dictionary = corpora.Dictionary(line.lower().split() for line in open('mycorpus.txt'))
# remove stop words and words that appear only once
stop_ids = [dictionary.token2id[stopword] for stopword in stoplist
             if stopword in dictionary.token2id]
once_ids = [tokenid for tokenid, docfreq in iteritems(dictionary.dfs) if docfreq == 1]
dictionary.filter_tokens(stop_ids + once_ids)  # remove stop words and words that appear only once
dictionary.compactify()  # remove gaps in id sequence after words that were removed
print(dictionary)

# gensim.models.wrappers.dtmmodel.DtmModel(dtm_path, corpus=None, time_slices=None, mode='fit', model='dtm', num_topics=100,
#  id2word=None, prefix=None, lda_sequence_min_iter=6, lda_sequence_max_iter=20, lda_max_em_iter=10, alpha=0.01, top_chain_var=0.005, rng_seed=0, initialize_lda=True)
model = models.wrappers.DtmModel('bin/dtm-win64.exe', corpus_memory_friendly, time_slices=[9], num_topics=20, id2word=dictionary)

topics = model.show_topics(num_topics=5, times=1, num_words=10)
for s in topics:
    print s