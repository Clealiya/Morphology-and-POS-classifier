# create dictionary of word and convert words to number with this dictionary

import json
from random import random, seed
from typing import Dict, List, Union

Word_Info = List[Union[str, int]]   # example: ['6', 'flights', 'flight', 'NOUN', '_', 'Number=Plur', '1', 'obj', '_', '_']
Sentence = List[Word_Info]          # list of word info
Sequence = List[Word_Info]          # list of word info with the same length

seed(0)

def create_vocab(data: List[Sentence], word_index: int, pad: str, unk: str) -> Dict[str, int]:
    """ create a vocabulary (python dict) witch match each word to an unique number
    data: list of Sentences
    index: index of the word in the word_info
    pad: pad caracter
    unk: unk caracter
    """
    vocab = {pad: 0, unk: 1}
    word_number = 2
  
    for sentence in data:
        for word_info in sentence:
            word = word_info[word_index]
            if word not in vocab:
                vocab[word] = word_number
                word_number += 1
    return vocab


def save_vocab(vocab: Dict[str, int], path: str) -> None:
    """ save the dictionary in the path"""
    with open(path, 'w', encoding='utf8') as f:
        json.dump(vocab, f)
        f.close()


def load_dictionary(filepath: str) -> Dict[str, int]:
    """ load the vocabulary from filepath (its a json file)"""
    with open(filepath, 'r', encoding='utf8') as f:
        vocab = json.load(f)
        f.close()
    return vocab


def replace_word2int(data: List[Sequence],
                     word_index: int,
                     vocab: Dict[str, int],
                     unk_rate: float,
                     unk: str
                     ) -> None:
    """ replace word in all Sequence by word number in the vocabulary: vocab
    sometime (with unk_rate), replace a word by unk caracter
    !!! Replace data by the new data and return nothing !!!
    """
    unk_index = vocab[unk]
    for sequence in data:
        for word_info in sequence:
            if word_info[word_index] in vocab and random() > unk_rate:
                word_info[word_index] = vocab[word_info[word_index]]
            else:
                word_info[word_index] = unk_index


if __name__ == '__main__':
    #load a dictionary
    vocab = load_dictionary("dictionary/French.json")
    print(vocab)