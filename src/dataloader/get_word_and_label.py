from typing import List, Callable, Tuple

from src.dataloader.convert_label import Convert

Word_Info = List[str]       # example: ['6', 'flights', 'flight', 'NOUN', '_', 'Number=Plur', '1', 'obj', '_', '_']
Sequence = List[Word_Info]  # list of word info with the same length


def split_data_to_word_label(data: List[Sequence],
                             word_index: int,
                             label_index: int,
                             convert_label: Convert,
                             del_data_after: bool
                             ) -> Tuple[List[List[int]], List[List[int]]]:
    """ split data to have word x and label y
    label str will be replace with the convert_label_function 
            like convert_pos or convert_morphy (see convert_label.py)"""
    len_word_info = len(data[0][0])
    assert word_index < len_word_info, f"Error, word_index={word_index} is higher that len(word_info):{len_word_info}"
    x, y = [], []
    for sequence in data:
        seq_x, seq_y = [], []
        for word_info in sequence:
            seq_x.append(word_info[word_index])
            seq_y.append(convert_label.encode(label_to_convert=word_info[label_index]))
        x.append(seq_x)
        y.append(seq_y)

    if del_data_after:
        del data
    
    return x, y
