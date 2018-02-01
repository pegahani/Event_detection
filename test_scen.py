import tools
from evaluation import compare_with_AMI_results
from tf_idf import tf_idf_class

file = open('./result_scen.txt', 'w')
adress = './manuel_corpus/'

# #
# #

meet_list = tools.scenario_based

file.write('************************' + str(meet_list) + '*************************************\n')
all_documents = tools.all_scenario_based_documents
example = tf_idf_class(all_documents)
tfidf_representation = example.tfidf()

for meet in meet_list:
        file.write(meet+ '\n')
        file.write('*****************'+ '\n')

        c = compare_with_AMI_results(meet)
        c.get_resumes()

        candidate = c.get_best_k_tfidf(20, tfidf_representation[tools.scenario_based.index(meet)])
        file.write('candidate = ' + str(candidate)+ '\n')

        bleu_measure = c.bleu_evaluation(candidate)

        file.write('score_abstractive = ' + str(bleu_measure[0]) + '\n')
        file.write(' score_extractive = ' + str(bleu_measure[1])+ '\n')

        file.write('\n')

        file.flush()

file.close()