import os
import sys
import yaml
from numpy import prod
from icecream import ic 
from easydict import EasyDict
from typing import Mapping, Any, List, Union, Iterator, Tuple, Optional
from os.path import dirname as up

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

sys.path.append(up(os.path.abspath(__file__)))
sys.path.append(up(up(os.path.abspath(__file__))))
sys.path.append(up(up(up(os.path.abspath(__file__)))))

from config.process_config import process_config
from src.model.LSTM import LSTMClassifier


NUN_C_POSSIBILITY = [2, 2, 3, 5, 3, 4, 13, 2, 13, 5, 5, 5, 5, 2, 4, 2, 3, 4, 6, 3, 4, 5, 2, 3, 2, 2, 3, 4]



def get_pos_model(config: EasyDict) -> torch.nn.Module:
    """ get model according a configuration """

    task_name = config.task.task_name
    num_classes = config.task[f'{task_name}_info'].num_classes
    morphy_config = config.model.lstm_morphy

    if task_name == "get_pos":
        pos_config = config.model.lstm_pos
        model = LSTMClassifier(num_words=config.data.vocab.num_words,
                               embedding_size=pos_config.embedding_size,
                               lstm_hidd_size_1=pos_config.lstm_hidd_size_1,
                               lstm_hidd_size_2=pos_config.lstm_hidd_size_2,
                               fc_hidd_size=pos_config.fc_hidd_size,
                               num_classes=num_classes,
                               bidirectional=pos_config.bidirectional,
                               activation=pos_config.activation,
                               num_c_possibility=1,
                               dropout=pos_config.dropout)

    return model


def load_checkpoint(model: torch.nn.Module,
                    experiment_path: str,
                    device: Optional[torch.device]='cpu') -> None:
    """ load checkpoint from logging_path in the model """
    weight_path = os.path.join(experiment_path, 'checkpoint.pt')

    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"weigth was not found in {weight_path}")
    
    model.load_state_dict(torch.load(weight_path, map_location=device))

def load_config(path: str) -> EasyDict:
    stream = open(path, 'r')
    return EasyDict(yaml.safe_load(stream))

class MorphPosLSTMClassifier(nn.Module):
    def __init__(self,
                 num_words: int,
                 embedding_size: int,
                 lstm_hidd_size_1: int,
                 num_classes: int,
                 num_c_possibility: List[int],
                 lstm_hidd_size_2: Union[int, None]=None,
                 fc_hidd_size: List[int]=[], 
                 bidirectional: bool=True,
                 activation: str='relu',
                 dropout: float=0,
                 add_zero: bool=False,
                 pos_config: str="logs/get_pos_French_2"
                 ) -> None:
    
        """ Model LSTM 
        ## Arguments:
        num_words: int
            number of word in the vocabulary
        embedding_size: int
            size of embedding
        lstm_hidd_size_1: int
            size of the first lstm layer
        num_classes: int
            number of classes

        ## Optional Arguments:
        lstm_hidd_size_2: int or None = None
            None -> not a second lstm layer; 
            int -> size of the second lstm layer
        fc_hidd_size: list of int = []
            List of size of Dense layers.
            !!! Not count the last layer in fc_hidd_size witch give the number of classes !!!
        bidirectional: bool = true
            Does the lstm layers go in both direction
        activation: str = relu
            choose an activation function between relu or softmax
        num_c_possibility: int = 1
            number of features of classes. must be 1 for get_pos and not 1 for get_mophy
        """
        self.num_c_possibility = num_c_possibility
        self.max_num_possibility = max(num_c_possibility)
        self.add_zero = add_zero

        super(MorphPosLSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=num_words,
                                      embedding_dim=embedding_size)
        
        # LSTM Layers
        mul = 2 if bidirectional else 1
        self.lstm_1 = nn.LSTM(input_size=embedding_size,
                              hidden_size=lstm_hidd_size_1,
                              batch_first=True,
                              bidirectional=bidirectional)
        
        self.dropout = nn.Dropout(p=dropout)

        self.have_lstm_2 = lstm_hidd_size_2 is not None
        if self.have_lstm_2:
            self.lstm_2 = nn.LSTM(input_size=lstm_hidd_size_1 * mul,
                                  hidden_size=lstm_hidd_size_2,
                                  batch_first=True,
                                  bidirectional=bidirectional)
        
        assert activation in ['relu', 'sigmoid'], f"Error, activation must be relu or sigmoid but found '{activation}'"
        self.activation = nn.ReLU() if activation == 'relu' else nn.Sigmoid()

        # Fully Connected Layers
        last_lstm_hidd_layers_size = lstm_hidd_size_1 if not self.have_lstm_2 else lstm_hidd_size_2
        fc_hidd_size = [last_lstm_hidd_layers_size * mul+19] + fc_hidd_size #on rajoute 19 car on concatène les prédictions de pos concaténées !!!
        self.fc = []

        for i in range(len(fc_hidd_size) - 1):
            self.fc.append(self.dropout)
            self.fc.append(self.activation)
            self.fc.append(nn.Linear(in_features=fc_hidd_size[i],
                                     out_features=fc_hidd_size[i + 1]))
        self.fc = nn.Sequential(*self.fc)

        self.morph: List[nn.Linear] = []
        for i in range(num_classes):
            if add_zero:
                self.morph.append(nn.Linear(in_features=fc_hidd_size[-1],
                                            out_features=self.num_c_possibility[i]))
            else:
                self.morph.append(nn.Linear(in_features=fc_hidd_size[-1],
                                            out_features=self.max_num_possibility))
            
        self.num_classes = num_classes

        #on récupère le modèle de pos pré-entrainé
        config = load_config(os.path.join(pos_config,'config.yaml'))
        process_config(config)
        model = get_pos_model(config)
        #load weights
        load_checkpoint(model,pos_config,device='cuda')
        self.pos_model = model

        for param in self.pos_model.parameters():
            param.requires_grad = False
        
    
        print("model loaded:", model)

    def forward(self, x: Tensor) -> Tensor:
        """ forward
        take x Tensor with shape: (B, K)
        return output Tensor with shape: (B, K, C)

        where:
            B: batch_size
            K: sequence size
            C: number of classes
        """

        B, K = x.shape

        pos_predictions = self.pos_model.forward(x) #fait la prédiction à partir du modèle de pos pré-entrainé
        #print("pos_predictions:", pos_predictions)  #on obtient (2048, 10, 19) comme shape
        # Apply softmax to get probabilities
        probabilities = F.softmax(pos_predictions, dim=-1)

        # Apply threshold to get 1 or 0 values. threshold =1/nb_classes de pos pas de morphy !!
        threshold = 1/19
        binary_predictions = (probabilities > threshold).float()

        #print("binary_predictions:", binary_predictions)
        pos_predictions=binary_predictions

        x = self.embedding(x)

        #print("x_prim.shape:", x_prim.shape) #on obtient (2048, 10, 64)
   

        x = self.activation(x)

        x, _ = self.lstm_1(x) #x.shape = (B, K, lstm_hidd_size_1 * mul)

        if self.have_lstm_2:
            x = self.dropout(x)
            x = self.activation(x)
            x, _ = self.lstm_2(x)


        #print shape of x
        #print("out of lstm shape:", x.shape) #on obtient (2048, 10, 256)


     

        # Concatenate predictions from POS model
        x = torch.cat((x, pos_predictions), dim=-1) #x.shape = (2048, 10, 275)

        x = self.fc(x)

        x = self.dropout(x)
        x = self.activation(x)
        # x = x.view(-1, K, self.num_classes, self.max_num_possibility)
        logit_list = []
        for i in range(self.num_classes):
            logits_i = self.morph[i](x)
            if self.add_zero:
                padding = torch.full((B, K, self.max_num_possibility - self.num_c_possibility[i]),
                                     fill_value=-1000,
                                     dtype=torch.float32,
                                     device=x.device)
                # zeros = torch.zeros((B, K, self.max_num_possibility - self.num_c_possibility[i]),
                #                     dtype=torch.float32,
                #                     device=x.device)
                logit_list.append(torch.concat([logits_i, padding], dim=2))
            else:
                logit_list.append(logits_i)
        logits = torch.stack(logit_list, dim=2)
            
        return logits

    def get_number_parameters(self) -> int:
        """ return the number of parameters of the model """
        return sum([prod(param.size()) for param in self.parameters()])
   
    def named_parameters(self, prefix: str = '', recurse: bool = True, remove_duplicate: bool = True) -> Iterator[Tuple[str, Parameter]]:
        yield from super().named_parameters(prefix=prefix, recurse=recurse, remove_duplicate=remove_duplicate)
        for c in range(self.num_classes):
            yield from self.morph[c].named_parameters(prefix=prefix, recurse=recurse, remove_duplicate=remove_duplicate)
    
    def to(self, device: torch.device):
        self = super().to(device)
        for c in range(self.num_classes):
            self.morph[c] = self.morph[c].to(device)
        return self
    
    def state_dict(self):
        output = {}
        param_in_list: dict[str, List[Tensor]] = {'weight': [], 'bias': []}

        for name, param in self.named_parameters():
            param = param.to('cpu')
            if name not in ['weight', 'bias']:
                output[name] = param
            else:
                param_in_list[name].append(param)
        
        for name, value in param_in_list.items():
            for i in range(len(value)):
                output[f'{name}_{i}'] = value[i]
        
        return output

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = False):
        error = super().load_state_dict(state_dict, strict=False)
        ic(error)
        if error.missing_keys != []:
            raise ValueError('erreur de corespondance entre les poids')

        if len(error.unexpected_keys) != 2 * self.num_classes:
            raise ValueError('erreur de corespondance entre les poids')

        for i in range(self.num_classes):
            self.morph[i].weight = torch.nn.Parameter(state_dict[f'weight_{i}'])
            self.morph[i].bias = torch.nn.Parameter(state_dict[f'bias_{i}'])


if __name__ == '__main__':



    model = MorphPosLSTMClassifier(num_words=67814,
                                   embedding_size=64,
                                   lstm_hidd_size_1=64,
                                   lstm_hidd_size_2=128,
                                   fc_hidd_size=[128, 64],
                                   num_classes=28,
                                   bidirectional=True,
                                   activation='relu',
                                   num_c_possibility=NUN_C_POSSIBILITY,
                                   dropout=0.1,
                                   add_zero=True,
                                   pos_config="logs/get_pos_French"
                                   )
    
    print(model)

    for name, param in model.state_dict().items():
        print(name, param.shape, param.device, param.requires_grad)

    model.load_state_dict(model.state_dict())

    print(model.get_number_parameters())

    # x = torch.randint(low=0, high=67814, size=(2048, 10))

    # print(x.shape, x.device)

    # y = model.forward(x=x)
    # print(y.shape)

    
