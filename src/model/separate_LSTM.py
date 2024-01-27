import torch
from torch import Tensor
import torch.nn as nn
from numpy import prod
from icecream import ic 

from typing import Any, List, Mapping, Union, Iterator, Tuple
from torch.nn.parameter import Parameter

NUN_C_POSSIBILITY = [2, 2, 3, 5, 3, 4, 13, 2, 13, 5, 5, 5, 5, 2, 4, 2, 3, 4, 6, 3, 4, 5, 2, 3, 2, 2, 3, 4]


class MorphLSTMClassifier(nn.Module):
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
                 add_zero: bool=False
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

        super(MorphLSTMClassifier, self).__init__()
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
        fc_hidd_size = [last_lstm_hidd_layers_size * mul] + fc_hidd_size
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

        x = self.embedding(x)
        x = self.activation(x)

        x, _ = self.lstm_1(x)

        if self.have_lstm_2:
            x = self.dropout(x)
            x = self.activation(x)
            x, _ = self.lstm_2(x)

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
        if error.missing_keys != []:
            raise ValueError('erreur de corespondance entre les poids')

        if len(error.unexpected_keys) != 2 * self.num_classes:
            raise ValueError('erreur de corespondance entre les poids')

        for i in range(self.num_classes):
            self.morph[i].weight = torch.nn.Parameter(state_dict[f'weight_{i}'])
            self.morph[i].bias = torch.nn.Parameter(state_dict[f'bias_{i}'])



if __name__ == '__main__':
    model = MorphLSTMClassifier(num_words=67814,
                                embedding_size=64,
                                lstm_hidd_size_1=64,
                                lstm_hidd_size_2=None,
                                fc_hidd_size=[64],
                                num_classes=28,
                                bidirectional=True,
                                activation='relu',
                                num_c_possibility=NUN_C_POSSIBILITY,
                                dropout=0.1,
                                add_zero=False)
    
    print(model)

    for name, param in model.named_parameters():
        print(name, param.shape, param.device)

    print(model.get_number_parameters())

    # x = torch.randint(low=0, high=67814, size=(2048, 10))

    # print(x.shape, x.device)

    # y = model.forward(x=x)
    # print(y.shape)

    checkpoint = model.state_dict()

    for name, param in checkpoint.items():
        print(name, param.shape, param.device)
    
    model.load_state_dict(state_dict=checkpoint, strict=False)
    
    
