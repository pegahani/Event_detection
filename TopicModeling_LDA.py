#https://rstudio-pubs-static.s3.amazonaws.com/79360_850b2a69980c4488b1db95987a24867a.html

import gensim
import numpy as np
import os

import lda

from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models


import tools
from tools import tokenize

class TM_LDA:
    def __init__(self, all_documents, num_topics, passes, num_words):

        self.all_documents = all_documents
        self.num_topics = num_topics
        self.passes = passes
        self. num_words =  num_words

        self.file = open('./result/Topic_modeling_LDA.txt', 'w')

    def prepare_texts(self):
        tokens = [tokenize(d) for d in self.all_documents]
        # create English stop words list
        en_stop = get_stop_words('en')
        en_stop = en_stop + [u'yeah' , u'um' , u'uh' , u'just' , u'like' , u'okay' , u'know' , u'well' , u'that' , u'right' ,
                             u'can', u'mmhmm' , u'thing', u'dont' , u'oh' , u'get' , u'mm' , u'mayb' , u'will' , u'ye', u'things'
                             , u'actually', u'anyway', u'anywhere', u'doesnt', u'also', u'maybe', u'thats', u'im', u'someth', u'gonna', u'want', u'think', u'one']

        # remove stop words from tokens
        stopped_tokens = [[i for i in corpus if not i in en_stop] for corpus in tokens]
        #print 'stopped_tokens', stopped_tokens

        # Create p_stemmer of class PorterStemmer
        p_stemmer = PorterStemmer()

        # stem token
        texts = [[p_stemmer.stem(i) for i in corpus] for corpus in stopped_tokens]

        return texts

    def find_lda_topics(self):
        self.texts = self.prepare_texts()
        self.dictionary = corpora.Dictionary(self.texts)

        corpus = [self.dictionary.doc2bow(text) for text in self.texts]

        self.ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=self.num_topics, id2word=self.dictionary, passes= self.passes)

        self.file.write('topics                         words'+ '\n')
        for topic in self.ldamodel.print_topics(num_topics=self.num_topics, num_words=self.num_words):
            self.file.write(str(topic[0]) + '     '+ str(topic[1]) + '\n')
            self.file.flush()

        return

    def get_max_related_topic(self, input):

        which_topic = -1
        max = -1.0

        for elem in input:
            if elem[1] > max:
                max = elem[1]
                which_topic = elem[0]

        return which_topic

    def show_documenrs_vs_topics(self,meet_list):
        counter = 0

        self.file.write('\n')
        self.file.write('******************************************************\n')

        self.file.write('corpus\t')
        self.file.write('max topics\t')
        for i in range(self.num_topics):
            self.file.write( 'topic'+ str(i+1) +  'prob \t')

        self.file.write('\n')

        for d in self.texts:
            bow = self.dictionary.doc2bow(d)
            t = self.ldamodel.get_document_topics(bow = bow,  minimum_probability= 0.0)

            self.file.write(meet_list[counter] +  '   ' + str(self.get_max_related_topic(t)) + '      ' + str(t[0][1]) +
                            '      ' + str(t[1][1]) + '      ' + str(t[2][1]) + '\n') #+ '      ' + str(t))
            counter += 1

            self.file.flush()

        return

# CORPUS_PATH = './test'
# filenames = sorted([os.path.join(CORPUS_PATH, fn) for fn in os.listdir(CORPUS_PATH)])
#
# # files are located in ./test/
# filenames = filenames[0:-3]
#
# meet_list = ['meet_1', 'meet_2']
# adress_ = './test/'
# all_documents = [tools.text_to_string(adress_ + i + '.txt') for i in meet_list]
#
# lda_instance = TM_LDA(all_documents, num_topics= 3, passes= 20,num_words= 10)
# lda_instance.find_lda_topics()
#
# lda_instance.show_documenrs_vs_topics(meet_list = meet_list)


