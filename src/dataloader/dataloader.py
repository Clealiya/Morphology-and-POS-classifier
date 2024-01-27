import os
from easydict import EasyDict
from typing import List, Tuple, Dict

import torch
from torch.utils.data import Dataset, DataLoader
import sys
sys.path.append(os.path.join(sys.path[0], '..'))
sys.path.append(os.path.join(sys.path[0], '..', '..'))

from src.dataloader import get_sentences
from src.dataloader import get_sequences
from src.dataloader import vocabulary
from src.dataloader import convert_label
from src.dataloader import get_word_and_label


Word_Info = List[str]       # example: ['6', 'flights', 'flight', 'NOUN', '_', 'Number=Plur', '1', 'obj', '_', '_']
Sequence = List[Word_Info]  # list of word info with the same length
Tensor = torch.Tensor       # Torch Tensor


class DataGenerator(Dataset):
    def __init__(self, config: EasyDict, mode: str) -> None:
        assert mode in ['train', 'val', 'test'], f"Error, expected mode is train, val or test but found '{mode}'"
        self.mode = mode

        self.indexes = config.data.indexes
        self.word_index = get_sentences.get_word_index_in_indexes(self.indexes)
        self.task = config.task.task_name
        self.label_index = self.get_label_index()
        self.num_classes = config.task[f'{self.task}_info'].num_classes

        self.c = self.num_classes if self.task == 'get_pos' else config.task.get_morphy_info.num_features

        data, self.vocab = get_data(cfg=config.data, mode=self.mode)

        convert = convert_label.POS() if self.task == 'get_pos' else convert_label.Morphy()

        self.x, self.y = get_word_and_label.split_data_to_word_label(data=data,
                                                                     word_index=self.word_index,
                                                                     label_index=self.label_index,
                                                                     convert_label=convert,
                                                                     del_data_after=True)
        
        self.x = torch.tensor(self.x).to(torch.long)
        self.y = torch.tensor(self.y).to(torch.long)

        assert len(self.x) == len(self.y), f"Error, len between word and label is not the same"
        self.num_data = len(self.x)

        print(f"the {self.mode} generator was created")        

    def __len__(self) -> int:
        return self.num_data

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        """ return word: x and label: y
        if task == get_pos:
            shape x: (B, K)
            shape y: (B, K)             # not a one hot
        
        if task == get_morphy:
            shape x: (B, K)
            shape y: (B, K, C, N)       # one hot

            where:  B: batch size
                    K: sequence length
                    C: number of classes (=28)
                    N: number of possibility by class (=13)
        """
        x = self.x[index]
        y = self.y[index]

        if self.task == 'get_morphy':
            y = torch.nn.functional.one_hot(y, num_classes=self.c).to(torch.float32)

        return x, y
    
    def get_label_index(self) -> int:
        assert self.task in ['get_pos', 'get_morphy'], f"Error, expected task is get_pos or get_morphy but found '{self.task}'."
        if self.task == 'get_pos':
            assert 3 in self.indexes, f"Error, if task=get_pos, 3 must be in config.data.indexes"
            return self.indexes.index(3)
        if self.task == 'get_morphy':
            assert 5 in self.indexes, f"Error, if task=get_morphy, 5 must be in config.data.indexes"
            return self.indexes.index(5)
    
    def get_vocab(self) -> Dict[str, int]:
        return self.vocab


def get_data(cfg: EasyDict, mode: str) -> Tuple[List[Sequence], Dict[str, int]]:
    """
    Retrieves the data for a given mode (train, test, or validation) based on the provided configuration.

    Args:
        cfg (EasyDict): The configuration object containing the necessary parameters.
        mode (str): The mode for which to retrieve the data: train, val or test

    Returns:
        List[Sequence]: The processed data in the form of sequences.
        Dict[str, int]: The vocabluary
    """
    folders = get_sentences.get_foldersname_from_language(datapath=cfg.path,
                                                          language=cfg.language)
    files = get_sentences.get_file_for_mode(folder_list=folders, mode=mode)
    data = get_sentences.get_sentences(files=files, indexes=cfg.indexes)

    word_index = get_sentences.get_word_index_in_indexes(cfg.indexes)
    vocab_path = os.path.join(cfg.vocab.path, cfg.language + '.json')
    
    if cfg.vocab.save and mode == 'train':
        vocab = vocabulary.create_vocab(data, word_index=word_index, pad=cfg.pad, unk=cfg.unk)
        vocabulary.save_vocab(vocab, path=vocab_path)
    else:
        vocab_path = os.path.join(cfg.vocab.path, cfg.language + '.json')
        vocab = vocabulary.load_dictionary(filepath=vocab_path)
    
    if mode == 'train':
        cfg.vocab.num_words = len(vocab)
        print(f"vocab size: {cfg.vocab.num_words}")
    
    sequence_function = get_sequences.find_sequence_function(cfg.sequence_function)
    data = get_sequences.create_sequences(sentences=data,
                                          sequence_function=sequence_function,
                                          k=cfg.sequence_length,
                                          pad=cfg.pad)
    
    vocabulary.replace_word2int(data,
                                word_index=word_index,
                                vocab=vocab,
                                unk_rate=cfg.vocab.unk_rate if mode == 'train' else 0,
                                unk=cfg.unk)

    return data, vocab


def create_dataloader(config: EasyDict, mode: str) -> Tuple[DataLoader, Dict[str, int]]:
    generator = DataGenerator(config=config, mode=mode)
    dataloader = DataLoader(dataset=generator,
                            batch_size=config.learning.batch_size,
                            shuffle=config.learning.shuffle,
                            drop_last=config.learning.drop_last)
    return dataloader, generator.get_vocab()


if __name__ == '__main__':
    # must to run dataloader like python .\src\dataloader\dataloader.py
    import yaml
    from icecream import ic 
    config = EasyDict(yaml.safe_load(open('config/config.yaml', 'r')))

    config.task.task_name = 'get_morphy'
    dataloader, _ = create_dataloader(config=config, mode='test')
    x, y = next(iter(dataloader))
    ic(x.shape, x.dtype)
    ic(y.shape, y.dtype)

    # ic(x[0])
    # ic(y[0])