import tools
from TopicModeling_NMF import TM_NMF

meet_list = tools.scenario_based + tools.non_scenario_based
adress = './manuel_corpus/'
all_documents = [adress + i + '.txt' for i in meet_list]

nmf_instance = TM_NMF(all_documents, num_topics=50 , num_top_words= 10, min_df=0, max_df= 20)
nmf_instance.find_NMF_topics()
nmf_instance.show_corpus_vs_topics()
nmf_instance.show_topic_words()