import os
import sys
import torch
import numpy as np
from icecream import ic
from easydict import EasyDict
from typing import List, Union

sys.path.append(os.path.join(sys.path[0], '..'))
sys.path.append(os.path.join(sys.path[0], '..', '..'))

from src.model.get_model import get_model
from src.dataloader import convert_label
from src.dataloader import get_sequences
from src.dataloader import vocabulary


def procces_data(config: EasyDict, txt_file: str) -> Union[List[str], torch.Tensor]:
    data = ''
    with open(file=txt_file, mode='r', encoding='utf8') as f:
        for line in f.readlines():
            data += line[:-1]
        f.close()
    
    data = data.split(' ')
    ic(data)

    sequences = get_sequences.dummy_sequences(sentence=data,
                                              k=config.data.sequence_length,
                                              pad=config.data.pad)
    vocab = vocabulary.load_dictionary(filepath=os.path.join(config.data.vocab.path, f'{config.data.language}.json'))

    x = []
    find_index = lambda word: vocab[word] if word in vocab else vocab[config.data.unk]
    for sequence in sequences:
        x.append(list(map(find_index, sequence)))
    x = torch.tensor(x, dtype=torch.int64)
    return data, x


def infer(config: EasyDict,
          logging_path: str,
          folder: str='infer') -> None:
    if torch.cuda.is_available() and config.learning.device == 'cuda':
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Get data
    data, x = procces_data(config=config, txt_file=os.path.join(folder, 'infer.txt'))
    x = x.to(device)
    ic(x)
    
    # Get model
    model = get_model(config)
    model = model.to(device)
    checkpoint_path = os.path.join(logging_path, 'checkpoint.pt')
    assert os.path.isfile(checkpoint_path), f'Error: model weight was not found in {checkpoint_path}'
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    ic(model)
    ic(model.get_number_parameters())

    model.eval()
    with torch.no_grad():
        y_pred = model.forward(x)
        y_pred = torch.argmax(y_pred, dim=-1)
        ic(y_pred.shape)

    y_pred = y_pred.to('cpu')
    output = decode_data(y_pred, task=config.task.task_name)
    save_infer(output, data, dst_path=os.path.join(folder, f'{config.name}_infer.txt'), pad=config.data.pad)
    

def decode_data(y_pred: torch.Tensor, task: str) -> np.ndarray:
    label_encoder = convert_label.get_convert(task=task)
    output = []
    B, K = y_pred.shape[:2]
    for b in range(B):
        s = []
        for k in range(K):
            if task == 'get_pos':
                label_to_decode = y_pred[b, k, ...].item()
            else:
                label_to_decode = list(map(lambda x: x.item(), y_pred[b, k, ...]))
            s.append(label_encoder.decode(label_to_decode))
        output.append(s)
    return np.array(output)


def save_infer(output: np.ndarray, data: List[str], dst_path: str, pad: str) -> None:
    B, K = output.shape
    final_pred = ''
    data = data + [pad] * (B * K - len(data))

    for b in range(B):
        for k in range(K):
            pred = f'{data[b * K + k]}\t: {output[b, k]}'
            print(pred)
            final_pred += pred + '\n'
    
    with open(file=dst_path, mode='w', encoding='utf8') as f:
        f.write(final_pred)
        f.close()
    
    print(f'infer was save in {dst_path}')
    


if __name__ == '__main__':
    import yaml
    logging_path = 'logs\get_morphy_sep0_French_0'
    config = EasyDict(yaml.safe_load(open(file=os.path.join(logging_path, 'config.yaml'))))
    infer(logging_path=logging_path, config=config, folder='infer')



