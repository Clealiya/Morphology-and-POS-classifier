import os
import json
from typing import List, Union


POS_CLASSES = ['PRON', 'AUX', 'VERB', 'DET', 'ADJ', 'NOUN', 'ADP', 'PROPN', 'NUM', 'CCONJ',
               '<PAD>', 'ADV', 'PART', 'INTJ', 'SYM', 'PUNCT', 'SCONJ', 'X', '_']


class Convert():
    def __init__(self) -> None:
        self.label = []
        self.num_classes = 0
    def encode(self, label_to_convert: str) -> Union[int, List[int]]:
        raise NotImplementedError
    def find_num_classes(self) -> int:
        return len(self.label)
    def decode(self, label_to_decode: Union[int, List[int]], remove_Not: bool=True) -> str:
        raise NotImplementedError
    

class POS(Convert):
    def __init__(self) -> None:
        super().__init__()
        self.label = POS_CLASSES
        self.num_classes = self.find_num_classes()

    def encode(self, label_to_convert: str) -> int:
        return self.label.index(label_to_convert)

    def decode(self, label_to_decode: int) -> str:
        if label_to_decode >= self.num_classes:
            raise ValueError(f'Expected label_to_decode be int between 0 and {self.num_classes} but found {label_to_decode}.')
        return self.label[label_to_decode]


class Morphy(Convert):
    def __init__(self) -> None:
        super().__init__()
        morphy = os.path.join('src', 'dataloader', 'morphy.json')
        with open(file=morphy, mode='r', encoding='utf8') as f:
            self.label: dict[str, List[str]] = json.load(f)
            f.close()
        self.num_classes = self.find_num_classes()
    
    def encode(self, label_to_convert: str) -> List[int]:
        label_to_convert = dict(map(lambda x: self.split_x(x=x), label_to_convert.split('|')))

        output = []
        for key, value in self.label.items():
            if key in label_to_convert:
                output.append(value.index(label_to_convert[key]))
            else:
                output.append(value.index('Not'))
        
        return output

    def split_x(self, x: str) -> List[str]:
        if x in ['_', '<PAD>']:
            return [x, 'Yes']
        return x.split('=')
    
    def decode(self,
               label_to_decode: List[int],
               remove_Not: bool=True) -> str:
        label_decode = {}
        for i, (key, value) in enumerate(self.label.items()):
            if label_to_decode[i] < len(value):
                label_decode[key] = value[label_to_decode[i]]
            else:
                label_decode[key] = value[0]
        
        output = ''
        for key, value in label_decode.items():
            if remove_Not and value == 'Not':
                pass
            else:
                output += f'{key}={value}|'
        
        if len(output) > 1:
            output = output[:-1]
        return output
            
            

def get_convert(task: str) -> Convert:
    if task == 'get_pos':
        convert = POS()
    if task == 'get_morphy':
        convert = Morphy()
    return convert
    

if __name__ == '__main__':
    from icecream import ic 

    pos_label = 'NOUN'
    convert = get_convert(task='get_pos')
    encode = convert.encode(label_to_convert=pos_label)
    ic(encode)
    decode = convert.decode(label_to_decode=encode)
    ic(decode)

    morphy_label = 'Emph=No|Number=Sing|Person=1|PronType=Prs'
    convert = get_convert(task='get_morphy')
    encode = convert.encode(label_to_convert=morphy_label)
    ic(encode)
    decode = convert.decode(label_to_decode=encode, remove_Not=True)
    ic(decode)