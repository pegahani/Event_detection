from __future__ import print_function
from __future__ import division
import math

#clean_tokenize = lambda doc: [d for d in doc if d != '']
from tools import tokenize


class tf_idf_class:
    def __init__(self, all_documents):#, corpus):
        #self.corpus = corpus
        self.all_documents = all_documents

    def materm_frequency(self,term, tokenized_document):
        """
        returns how many times term is repeated in tokenized_documents
        :param term:
        :param tokenized_document:
        :return:
        """
        return tokenized_document.count(term)

    def sublinear_term_frequency(self, term, tokenized_document):
        return 1 + math.log(tokenized_document.count(term))

    def augmented_term_frequency(self, term, tokenized_document):
        max_count = max([self.term_frequency(t, tokenized_document) for t in tokenized_document])
        return (0.5 + ((0.5 * self.term_frequency(term, tokenized_document)) / max_count))

    def inverse_document_frequencies(self,tokenized_documents):
        idf_values = {}
        all_tokens_set = set([item for sublist in tokenized_documents for item in sublist])
        for tkn in all_tokens_set:
            contains_token = map(lambda doc: tkn in doc, tokenized_documents)
            idf_values[tkn] = 1 + math.log(len(tokenized_documents) / (sum(contains_token)))
        return idf_values

    def tfidf(self):

        self.tokenized_documents = [tokenize(d) for d in self.all_documents]
        idf = self.inverse_document_frequencies(self.tokenized_documents)
        tfidf_documents = []

        for document in self.tokenized_documents:
        #print('***', self.corpus)
        #document = tokenize(self.corpus)

            doc_tfidf = {}
            for term in list(set(document)):  #idf.keys():
                print(term)
                tf = self.augmented_term_frequency(term, document)
                # doc_tfidf.append(tf * idf[term])
                doc_tfidf[term] = tf * idf[term]
            tfidf_documents.append(doc_tfidf)

        return tfidf_documents



