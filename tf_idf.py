from __future__ import print_function
from __future__ import division
import string
import math

tokenize = lambda doc: doc.lower().split(" ")

document_0 = "China has a strong economy that is growing at a rapid pace. However politically it differs greatly from the US Economy."
document_1 = "At last, China seems serious about confronting an endemic problem: domestic violence and corruption."
document_2 = "Japan's prime minister, Shinzo Abe, is working towards healing the economic turmoil in his own country for his view on the future of his people."
document_3 = "Vladimir Putin is working hard to fix the economy in Russia as the Ruble has tumbled."
document_4 = "What's the future of Abenomics? We asked Shinzo Abe for his views"
document_5 = "Obama has eased sanctions on Cuba while accelerating those against the Russian Economy, even as the Ruble's value falls almost daily."
document_6 = "Vladimir Putin was found to be riding a horse, again, without a shirt on while hunting deer. Vladimir Putin always seems so serious about things - even riding horses."


all_documents = [document_0, document_1, document_2, document_3, document_4, document_5, document_6]

tokenized_documents = [tokenize(d) for d in all_documents] # tokenized docs
all_tokens_set = set([item for sublist in tokenized_documents for item in sublist])

def term_frequency(term, tokenized_document):
    """
    returns how many times term is repeated in tokenized_documents
    :param term:
    :param tokenized_document:
    :return:
    """
    return tokenized_document.count(term)

def sublinear_term_frequency(term, tokenized_document):
    return 1 + math.log(tokenized_document.count(term))

def augmented_term_frequency(term, tokenized_document):
    max_count = max([term_frequency(t, tokenized_document) for t in tokenized_document])
    return (0.5 + ((0.5 * term_frequency(term, tokenized_document))/max_count))


def inverse_document_frequencies(tokenized_documents):

    idf_values = {}
    all_tokens_set = set([item for sublist in tokenized_documents for item in sublist])
    for tkn in all_tokens_set:
        contains_token = map(lambda doc: tkn in doc, tokenized_documents)
        idf_values[tkn] = 1 + math.log(len(tokenized_documents)/(sum(contains_token)))
    return idf_values

def tfidf(documents):

    tokenized_documents = [tokenize(d) for d in documents]
    print('tokenized_documents',tokenized_documents)
    idf = inverse_document_frequencies(tokenized_documents)
    print('idf', idf)
    tfidf_documents = []

    for document in tokenized_documents:
        doc_tfidf = []
        for term in document:#idf.keys():
            tf = augmented_term_frequency(term, document)
            doc_tfidf.append(tf * idf[term])
        tfidf_documents.append(doc_tfidf)

    return tfidf_documents

tfidf_representation = tfidf(all_documents)
print(tfidf_representation)
print(tfidf_representation[0])
print(tfidf_representation[1])
print(tfidf_representation[2])
print(tfidf_representation[3])
print(tfidf_representation[4])
print(tfidf_representation[5])
print(tfidf_representation[6])
print(len(tfidf_representation[6]))
print(all_tokens_set)
print(len(all_tokens_set))


