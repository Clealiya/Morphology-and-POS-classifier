import os
import sys
import json
from os.path import dirname as up
from typing import List

sys.path.append(up(os.path.abspath(__file__)))
sys.path.append(up(up(os.path.abspath(__file__))))
sys.path.append(up(up(up(os.path.abspath(__file__)))))


def load_dictionary(filepath: str=os.path.join('dictionary', 'baseline.json')) -> dict[str, str]:
    """ load the vocabulary from filepath (its a json file)"""
    with open(filepath, 'r', encoding='utf8') as f:
        vocab = json.load(f)
        f.close()
    vocab['<PAD>'] = '<PAD>=Yes'
    return vocab

def get_data(txt_file: str=os.path.join('infer', 'infer.txt')) -> List[str]:
    data = ''
    with open(file=txt_file, mode='r', encoding='utf8') as f:
        for line in f.readlines():
            data += line[:-1]
        f.close()
    
    data = data.split(' ')
    data += ['<PAD>'] * (10 - len(data) % 10)
    return data


def run_infer(vocab_path: str=os.path.join('dictionary', 'baseline.json'),
              txt_file: str=os.path.join('infer', 'infer.txt'),
              dst_path: str=os.path.join('infer','baseline_infer.txt'),
              ) -> None:
    
    vocab = load_dictionary(vocab_path)
    data = get_data(txt_file)

    output = ''
    unk_label = '_=Yes'
    for word in data:
        word_infer = vocab[word] if word in vocab else unk_label
        output += f'{word}\t: {word_infer}\n'

    with open(file=dst_path, mode='w', encoding='utf8') as f:
        f.write(output)
        f.close()
    
    print(f'save infer in {dst_path}')


if __name__ == '__main__':
    run_infer()
    