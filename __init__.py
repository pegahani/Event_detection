import tools
from evaluation import compare_with_AMI_results
from tf_idf import tf_idf_class

file = open('./result/result_4_4.txt', 'w')
adress = './manuel_corpus/'

# #
# #

classification_scenario_based = [['meet_1', 'meet_2', 'meet_3', 'meet_4'], ['meet_5', 'meet_6', 'meet_7', 'meet_8'], ['meet_10', 'meet_11',
                  'meet_12'], ['meet_13', 'meet_14', 'meet_15', 'meet_16'], ['meet_17', 'meet_18', 'meet_19', 'meet_20'],
                  ['meet_21', 'meet_22', 'meet_23'], ['meet_25', 'meet_26', 'meet_27', 'meet_28'], ['meet_29', 'meet_30',
                  'meet_31', 'meet_32'], ['meet_33', 'meet_34', 'meet_35', 'meet_36'], ['meet_37', 'meet_38', 'meet_39',
                  'meet_40'], ['meet_41', 'meet_42', 'meet_43', 'meet_44'], ['meet_45', 'meet_46', 'meet_47', 'meet_48'],
                  ['meet_49', 'meet_50', 'meet_51', 'meet_51b'], ['meet_52', 'meet_53', 'meet_54', 'meet_55'], ['meet_56',
                  'meet_57', 'meet_58', 'meet_59'], ['meet_60', 'meet_61', 'meet_62', 'meet_63'], ['meet_64', 'meet_65',
                  'meet_66', 'meet_67'], ['meet_68', 'meet_69', 'meet_70', 'meet_71'], ['meet_72', 'meet_73', 'meet_74',
                  'meet_75'], ['meet_76', 'meet_77', 'meet_78', 'meet_79'], ['meet_80', 'meet_81', 'meet_82', 'meet_83'],
                  ['meet_84', 'meet_85', 'meet_86', 'meet_87'], ['meet_88', 'meet_89', 'meet_90', 'meet_91'], ['meet_92',
                  'meet_93', 'meet_94', 'meet_95'], ['meet_96', 'meet_97', 'meet_98', 'meet_99'], ['meet_112', 'meet_113',
                  'meet_114', 'meet_115'], ['meet_116', 'meet_117', 'meet_118', 'meet_119'], ['meet_120', 'meet_121',
                  'meet_122', 'meet_123'], ['meet_124', 'meet_125', 'meet_126', 'meet_127'], ['meet_128', 'meet_129',
                  'meet_130', 'meet_131'], ['meet_132', 'meet_133', 'meet_134', 'meet_135'], ['meet_136', 'meet_137',
                  'meet_138', 'meet_139'], ['meet_140', 'meet_141', 'meet_142', 'meet_143'], ['meet_144', 'meet_145',
                  'meet_146', 'meet_147'], ['meet_148', 'meet_149', 'meet_150', 'meet_151']]

for meet_list in classification_scenario_based:
    file.write('************************' + str(meet_list) + '*************************************\n')
    all_documents = [tools.text_to_string(adress + i + '.txt') for i in meet_list]
    example = tf_idf_class(all_documents)  # , document_0)
    tfidf_representation = example.tfidf()

    for meet in meet_list:
        file.write(meet+ '\n')
        file.write('*****************\n')

        c = compare_with_AMI_results(meet)
        c.get_resumes()

        candidate = c.get_best_k_tfidf(20, tfidf_representation[meet_list.index(meet)])
        file.write('candidate = ' + str(candidate)+ '\n')

        bleu_measure = c.bleu_evaluation(candidate)

        file.write('score_abstractive = ' + str(bleu_measure[0])+ '\n')
        file.write(' score_extractive = ' + str(bleu_measure[1])+ '\n')
        file.flush()

    file.write('\n')
    file.flush()

file.close()
