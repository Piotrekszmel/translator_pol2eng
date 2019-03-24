from data_utils import *
from data_utils import save_clean_data
dataset_path = '/home/pszmelcz/Desktop/translator/raw_dataset/pol.txt'

en_sentences, pol_sentences = read_sentences(dataset_path)

text = load_data(dataset_path)
pairs = to_pairs(text)
clean_pairs = clean_pairs(pairs)
save_clean_data(clean_pairs, 'eng-pol.pkl')



dataset = load_clean_sentences('/home/pszmelcz/Desktop/translator/dataset/english-polish.pkl')
numpy.random.shuffle(dataset)
train, validation, test = dataset[:90000], dataset[90001:98000], dataset[98001:103596]

save_clean_data(dataset, 'english-polish_dataset.pkl')
save_clean_data(train, 'english-polish_train.pkl')
save_clean_data(validation, 'english-polish_validation.pkl')
save_clean_data(test, 'english-polish_test.pkl')