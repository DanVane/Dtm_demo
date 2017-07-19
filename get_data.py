# Import some modules for reading and getting data.
# If you don't have this modules, you must install them.
import csv
import MySQLdb
import re
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
import os
from gensim import corpora, models, similarities  # to create a dictionary

# Set years, this would be the timestamps
time_stamps = ['2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009']
# Set the conference name to be analyzed
# conference = ''


# DB MYSQL Connect. Put your credentials here.
db_host = 'localhost'  # Host
db_user = 'root'  # User
db_pass = '123456'  # Password
db_database = 'autoqa'  # Database

##Conect...
db = MySQLdb.connect(host=db_host, user=db_user, passwd=db_pass, db=db_database)

dat_outfile = open(os.path.join('data', 'metadata.dat'), 'w')
dat_outfile.write('id \t title \t abstract \t pub_year\n')  # write header

corpus_data = list()
# Set total_tweets list per year, starting at 0
total_data_list = [0 for year in time_stamps]

# Analyze each year..

time_stamps_count = 0

for year in time_stamps:  # For each year

    print('Analyzing year ' + str(year))

    # Set total_data to 0
    total_data = 0

    # Get data with mysql
    cursor = db.cursor()

    # Query
    query = "SELECT ID, TITLE, ABSTRACT, PUB_YEAR FROM paper WHERE ABSTRACT is not null and PUB_YEAR = " + year + " "

    # Execute query
    cursor.execute(query)
    result = cursor.fetchall()  # store results
    cursor.close()

    # For each result (data), get content and save it to the output file if it's not an empty line
    for line in result:

        # Remove @xxxx and #xxxxx
        content = [unicode(word.lower(), errors='ignore') for word in line[2].split() if
                   word.find('@') == -1 and word.find('#') == -1 and word.find('http') == -1]

        # join words list to one string
        content = ' '.join(content)

        # remove symbols
        content = re.sub(r'[^\w]', ' ', content)

        # remove stop words
        content = [word for word in content.split() if
                   word not in stopwords.words('english') and len(word) > 3 and not any(c.isdigit() for c in word)]

        # join words list to one string
        content = ' '.join(content)

        # Stemming and lemmatization
        lmtzr = WordNetLemmatizer()

        content = lmtzr.lemmatize(content)

        # Filter only nouns and adjectives
        tokenized = nltk.word_tokenize(content)
        classified = nltk.pos_tag(tokenized)

        content = [word for (word, clas) in classified if
                   clas == 'NN' or clas == 'NNS' or clas == 'NNP' or clas == 'NNPS' or clas == 'JJ' or clas == 'JJR' or clas == 'JJS']
        # join words list to one string
        content = ' '.join(content)

        if len(content) > 0:
            corpus_data.append([line[0], line[1], content, line[3]])
            total_data += 1
            dat_outfile.write(str(line[0]) + '\t' + line[1] + '\t' + str(line[3]) + '\t' + content)
            dat_outfile.write('\n')

    # Add the total data to the total data per year list
    total_data_list[time_stamps_count] += total_data

    time_stamps_count += 1

dat_outfile.close()  # Close the data file

# Write seq file
seq_outfile = open(os.path.join('data', 'foo-seq.dat'), 'w')
seq_outfile.write(str(len(total_data_list)) + '\n')  # number of TimeStamps

for count in total_data_list:
    seq_outfile.write(str(count) + '\n')  # write the total data per year (timestamp)

seq_outfile.close()

print('Done collecting data and writing seq')

# Transform each data content to a vector.

stoplist = set('for a of the and to in'.split())

# Construct the dictionary

dictionary = corpora.Dictionary(line[2].lower().split() for line in corpus_data)

# remove stop words and words that appear only once
stop_ids = [dictionary.token2id[stopword] for stopword in stoplist
            if stopword in dictionary.token2id]
once_ids = [tokenid for tokenid, docfreq in dictionary.dfs.iteritems() if docfreq == 1]
dictionary.filter_tokens(stop_ids + once_ids)  # remove stop words and words that appear only once
dictionary.compactify()  # remove gaps in id sequence after words that were removed

dictionary.save(os.path.join('data', 'dictionary.dict'))  # store the dictionary, for future reference

# Save vocabulary
vocFile = open(os.path.join('data', 'vocabulary.dat'), 'w')
for word in dictionary.values():
    vocFile.write(word + '\n')

vocFile.close()

print('Dictionary and vocabulary saved')


# Prevent storing the words of each document in the RAM
class MyCorpus(object):
    def __iter__(self):
        for line in corpus_data:
            # assume there's one document per line, tokens separated by whitespace
            yield dictionary.doc2bow(line[1].lower().split())


corpus_memory_friendly = MyCorpus()

multFile = open(os.path.join('data', 'foo-mult.dat'), 'w')

for vector in corpus_memory_friendly:  # load one vector into memory at a time
    multFile.write(str(len(vector)) + ' ')
    for (wordID, weigth) in vector:
        multFile.write('('+str(wordID) + ',' + str(weigth) + ' )')

    multFile.write('\n')

multFile.close()

print('Mult file saved')
