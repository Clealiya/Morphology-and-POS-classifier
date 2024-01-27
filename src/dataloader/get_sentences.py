# Open data and get each setence

import os
from typing import List, Union


Word_Info = List[str]       # example: ['6', 'flights', 'flight', 'NOUN', '_', 'Number=Plur', '1', 'obj', '_', '_']
Sentence = List[Word_Info]  # list of word info


def get_sentences(files: Union[str, List[str]], indexes: List[int]) -> List[Sentence]:
    """ get a list of Sentences of the file or a list of file.
    get only the information word[indexes[i]] for i in indexes
    indexes representation:
    0: id       1: word    2: lemma   3: pos   4: unk
    5: morphy   6: syntax  7: unk     8: unk   9: unk 
    """
    #TODO: mettre un *arg pour faire plus propre
    if type(files) != list:
        files = [files]
    data = []
    for file in files:
        with open(file=file, mode='r', encoding='utf8') as f:
            sequence = []
            for line in f.readlines():
                if len(line) <= 1 and len(sequence) != 0:
                    data.append(sequence)
                    sequence = []
                else:
                    if line[0] != '#':
                        line_split = line.split('\t')
                        sequence.append([line_split[i] for i in indexes])
        f.close()
        if len(sequence) != 0:
            data.append(sequence)
    return data


def get_foldersname_from_language(datapath: str, language: str) -> List[str]:
    """
    get the list of data folder witch each folder have language data 
    and at least 3 conllu file (for dev, train and test)
    """
    # get all folders
    all_folders = list(map(lambda folder: os.path.join(datapath, folder), os.listdir(datapath)))
    # get folder in language
    good_folder = list(filter(lambda foldername: language in foldername, all_folders))
    # fonction to get the number of conllu file inside a folder
    num_conllu_file = lambda folder: len([file for file in os.listdir(folder) if file.split('.')[-1] == 'conllu'])
    # fonction to know if te folder have exactly 3 conllu file (dev, train and test)
    have_enough_conllu_file = lambda folder: num_conllu_file(folder) == 3
    # get all the good folder 
    good_folder = list(filter(have_enough_conllu_file, good_folder))

    return good_folder
    
def get_file_for_mode(folder_list: List[str], mode: str) -> List[str]:
    """
    take a list of folder and a mode (train, val or test) 
    and give a list a file for the good mode
    """
    assert mode in ['train', 'val', 'test'], f"expected mode be train, val or test but found {mode}"
    name_mode = mode if mode in ['train', 'test'] else 'dev'

    files = []
    for folder in folder_list:
        for file in os.listdir(folder):
            if name_mode in file and file.split('.')[-1] == 'conllu':
                files.append(os.path.join(folder, file))
    return files


def get_word_index_in_indexes(indexes: List[int]) -> int:
    assert 1 in indexes, f"Error, 1 is not in indexes (1: word)"
    return indexes.index(1)


def get_all_label():
    from tqdm import tqdm
    from icecream import ic 
    import json

    data_path = os.path.join(os.getcwd(), 'data')

    all_label = []
    for language in tqdm(['English', 'French'], desc='get all data'):
        for mode in ['train', 'val', 'test']:
            folders = get_foldersname_from_language(datapath=data_path, language=language)
            files = get_file_for_mode(folder_list=folders, mode=mode)
            data = get_sentences(files=files, indexes=[1, 5])
            for sequence in data:
                all_label += list(map(lambda x: x[1], sequence))
    
    print(all_label[:10])
    print(len(all_label))


    not_value = 'Not'
    label_dict = {'_': [not_value, 'Yes'], '<PAD>': [not_value, 'Yes']}
    all_label = list(set(all_label))

    for label in tqdm(all_label, desc='get all label'):
        if label != '_':
            labels = label.split('|')
            labels = list(map(lambda x: x.split('='), labels))
            for key, value in labels:
                if key not in label_dict:
                    label_dict[key] = [not_value, value]
                else:
                    if value not in label_dict[key]:
                        label_dict[key].append(value) 
    
    ic(label_dict)
    with open(file='morphy2.json', mode='w') as f:
        json.dump(label_dict, f, indent=True)
        f.close()
    print('Done')


if __name__ == '__main__':
    folders = get_foldersname_from_language(datapath='data', language='French')
    files = get_file_for_mode(folder_list=folders, mode='test') + get_file_for_mode(folder_list=folders, mode='train') + get_file_for_mode(folder_list=folders, mode='val') 
    data = get_sentences(files=files, indexes=[1])
    count = 0
    words = []
    print(f'{len(data) = }')
    for sequence in data:
        count += len(sequence)
        for word in sequence:
            words.append(word[0])
    print(f'{count = }')
    print(f'nombre de mot unique {len(set(words))}')

