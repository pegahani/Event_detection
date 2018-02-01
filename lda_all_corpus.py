import tools
from TopicModeling_LDA import TM_LDA

meet_list = tools.scenario_based + tools.non_scenario_based
all_documents = tools.all_scenario_based_documents + tools.all_nonscenario_based_documents


lda_instance = TM_LDA(all_documents, num_topics= 10, passes= 500, num_words= 10)
lda_instance.find_lda_topics()

lda_instance.show_documenrs_vs_topics(meet_list = meet_list)

