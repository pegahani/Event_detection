import re

tokenize = lambda doc: [d for d in doc.lower().split(" ") if d != '']

def clean_line(data):

    data = data.replace('A : ', '')
    data = data.replace('B : ', '')
    data = data.replace('C : ', '')
    data = data.replace('D : ', '')

    data = re.sub(r'[^a-zA-Z0-9 ]',r'', data)

    return data

def text_to_string(input_file, is_resume_abstract = None, is_resume_extractive = None):  # meet_1
        #with open('./manuel_corpus/' + input_file , 'r') as myfile:
        #with open('./test/' + input_file , 'r') as myfile:

        data = ''

        if is_resume_extractive == None and is_resume_abstract == None:
            with open(input_file, 'r') as myfile:
                data = myfile.read().replace('\n', '')
                data = clean_line(data)

        if is_resume_abstract == True:
            with open(input_file, 'r') as myfile:
                data = myfile.read().replace('\n', ' ')

                data = data.replace('abstract : ', '')
                data = data.replace('actions : ', '')
                data = data.replace('decisions: ', '')
                data = data.replace('problems: ', '')

                data = clean_line(data)

        return data