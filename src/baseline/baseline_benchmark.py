import os
import sys
from os.path import dirname as up

sys.path.append(up(os.path.abspath(__file__)))
sys.path.append(up(up(os.path.abspath(__file__))))
sys.path.append(up(up(up(os.path.abspath(__file__)))))

import torch
import torch.nn.functional as F
import numpy as np
from icecream import ic
from easydict import EasyDict
import baseline_dictionary
from src.metrics import MOR_Metrics
from src.dataloader import get_sentences, convert_label, get_sequences


def test_dictionary() -> dict[str,str]:
    """
    Crée un dictionnaire des mots de toutes les phrases de données de test associés à leur premier label rencontré
    """
    folders = get_sentences.get_foldersname_from_language(datapath="data", language="French")
    files = get_sentences.get_file_for_mode(folder_list=folders, mode="test")
    data = get_sentences.get_sentences(files=files, indexes=[1, 5])  # indexes for morphy

    dico_train = baseline_dictionary.create_dictionnaire(data_path="data", language="French", mode="train")
    dico_train['<PAD>'] = '<PAD>=Yes'
    
    config = {'data': {'sequence_length': 10},
              'task': {'get_morphy_info': {'num_classes': 28},
                       'task_name': 'get_morphy'},
              'metrics': {'acc': True, 'allgood': True} }
    
    data = get_sequences.create_sequences(sentences=data, sequence_function=get_sequences.dummy_sequences, k=config['data']['sequence_length'], pad='<PAD>')
    metrics = MOR_Metrics(config=EasyDict(config))
    metrics_name = metrics.get_metrics_name()
    baseline_metrics = np.zeros((len(metrics_name)))

    label_encoder = convert_label.Morphy()
    unk_label = '_=Yes'
    n = 0

    for sentence in data:
        n += 1 
        y_pred_sequence = []
        y_true_sequence = []
        for word, y_true in sentence:
            
            y_pred = dico_train[word] if word in dico_train else unk_label
            y_pred_sequence.append(label_encoder.encode(label_to_convert=y_pred))
            y_true_sequence.append(label_encoder.encode(label_to_convert=y_true))

        y_true_encoding = torch.tensor([y_true_sequence])
        y_pred_encoding = torch.tensor([y_pred_sequence])
        num_classes = 13
        y_true_encoding = F.one_hot(y_true_encoding, num_classes=num_classes)
        y_pred_encoding = F.one_hot(y_pred_encoding, num_classes=num_classes)
        # y_pred et y_true de shape 1, 10, 28, 13
        baseline_metrics += metrics.compute(y_true=y_true_encoding, y_pred=y_pred_encoding)
    baseline_metrics = baseline_metrics / n

    ic(baseline_metrics)
    for i in range(len(metrics_name)):
        print(f"{metrics_name[i]} :\t{baseline_metrics[i]:.3f}")
    return baseline_metrics


if __name__ == "__main__":
    test_dictionary()