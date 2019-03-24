import re
import string
import unicodedata
import numpy as np
import numpy.random 
from pickle import load, dump
                
def read_sentences(file_path):              #reading dataset sentences and splitting into english and polish lists
        with open(file_path, 'r') as f:
                en_sentences = []
                pol_sentences = []
                for row in f:
                        row = row.rstrip('\n').split('\t')
                        en_sentences.append(row[0])
                        pol_sentences.append(row[1])
        
        return (en_sentences, pol_sentences)

def load_data(file_path):                  #loading data
        file = open(file_path, 'rt', encoding='utf-8')
        text = file.read()
        file.close()
        return text        

def to_pairs(text):                        #similar to read_sentences. Splitting into pairs
        lines = text.strip().split('\n')
        pairs = [line.split('\t') for line in lines]
        return pairs

def clean_pairs(lines):                    #cleaning samples
        cleaned = list()
        re_print = re.compile('[^%s]' % re.escape(string.printable)) #regex for filtering
        table = str.maketrans('','', string.punctuation) #mapping punctuation to None
        for pair in lines:
                clean_pair = list()
                for line in pair:
                        #line = line.replace('ą', 'a').replace('ć', 'c').replace('ę', 'e').replace('ł', 'l').replace('ń', 'n').replace('ó', 'o').replace('ś', 's').replace('ź', 'z').replace('ż', 'z')
                        line = line.replace('ą', 'a').replace('ć', 'c').replace('ę', 'e').replace('ł', 'l').replace('ń', 'n').replace('ó', 'o').replace('ś', 's').replace('ź', 'z').replace('ż', 'z')
                        line = unicodedata.normalize('NFD', line).encode('ascii', 'ignore')
                        line = line.decode('UTF-8')
                        line = line.split() #tokenize 
                        line = [word.lower() for word in line] #lowercase
                        line = [word.translate(table) for word in line] #remove punctuation
                        line = [re_print.sub('', w) for w in line] #remove non_printable chars
                        line = [word for word in line if word.isalpha()]
                        clean_pair.append(' '.join(line))
                cleaned.append(clean_pair)
        cleaned = np.asarray(cleaned)
        return cleaned


def save_clean_data(data, filename):            #saving clean data
        dump(data, open(filename, 'wb'))
        print('Saved {}'.format(filename))

def load_clean_sentences(filename):             #loading clean data
        return load(open(filename, 'rb'))

