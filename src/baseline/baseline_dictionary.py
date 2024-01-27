import os
import sys
from os.path import dirname as up

sys.path.append(up(os.path.abspath(__file__)))
sys.path.append(up(up(os.path.abspath(__file__))))
sys.path.append(up(up(up(os.path.abspath(__file__)))))

from typing import List
from icecream import ic
from easydict import EasyDict
import src.dataloader.vocabulary as vocabulary

import src.dataloader.get_sentences as get_sentences

Word_Info = List[str]       # example: ['6', 'flights', 'flight', 'NOUN', '_', 'Number=Plur', '1', 'obj', '_', '_']
Sequence = List[Word_Info]  # list of word info with the same length
Sentence = List[Word_Info]  # list of word info


def create_dictionnaire(data_path: str, language: str, mode: str) -> dict[str,str]:
    """
    Crée un dictionnaire des mots de toutes les phrases de data associés à leur premier label rencontré
    """
    folders = get_sentences.get_foldersname_from_language(datapath=data_path, language=language)
    files = get_sentences.get_file_for_mode(folder_list=folders, mode=mode)
    data = get_sentences.get_sentences(files=files, indexes=[1, 5])  # indexes for morphy

    dico = {}
    for sentence in data:
        for word in sentence:
            if word[0] not in dico:
                dico[word[0]] = word[1]
    vocabulary.save_vocab(dico, "dictionary/baseline.json")
    return dico


def prediction(dico: dict[str,str], sequence: list[Sentence]) -> list[str]:
    """
    Réalise une prédiction des classes des mots d'une séquence de sentences, 
    en attribuant au mot la classe correspondante dans le dictionnaire. 
    Si le mot est OOV, alors la classe attribuée est UNK.
    """
    prediction = []
    for sentence in sequence:
        for word in sentence:
            if word in dico:
                prediction.append(dico[word])

            else:
                prediction.append("<UNK>")
    return prediction


def launch_baseline(config: EasyDict) -> None:
    """
    Lance le modèle baseline
    """
    create_dictionnaire(data_path=config.data.path, language=config.data.language, mode=config)


if __name__ == "__main__":
    config = EasyDict()
    launch_baseline(config=config)