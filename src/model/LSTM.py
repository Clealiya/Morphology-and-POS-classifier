import torch
from torch import Tensor
import torch.nn as nn
from numpy import prod


from typing import List, Union, Optional


class LSTMClassifier(nn.Module):
    def __init__(self,
                 num_words: int,
                 embedding_size: int,
                 lstm_hidd_size_1: int,
                 num_classes: int,
                 lstm_hidd_size_2: Optional[Union[int, None]]=None,
                 fc_hidd_size: Optional[List[int]]=[], 
                 bidirectional: Optional[bool]=True,
                 activation: Optional[str]='relu',
                 num_c_possibility: Optional[int]=1,
                 dropout: Optional[float]=0
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
        dropout: float = 0.
            dropout rate between layers
        """
        super(LSTMClassifier, self).__init__()
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
        fc_hidd_size = [last_lstm_hidd_layers_size * mul] + fc_hidd_size + [num_classes * num_c_possibility]
        self.fc = []

        for i in range(len(fc_hidd_size) - 1):
            self.fc.append(self.dropout)
            self.fc.append(self.activation)
            self.fc.append(nn.Linear(in_features=fc_hidd_size[i],
                                     out_features=fc_hidd_size[i + 1]))
        self.fc = nn.Sequential(*self.fc)

        self.do_morphy = (num_c_possibility != 1)
        self.num_classes = num_classes
        self.num_c_possibility = num_c_possibility

    def forward(self, x: Tensor) -> Tensor:
        """ forward
        take x Tensor with shape: (B, K)
        return output Tensor with shape: (B, K, C)
        
        if morphy: output shape: (B, K, C, N)

        where:
            B: batch_size
            K: sequence size
            C: number of classes
        """
        sequence_length = x.shape[-1]

        x = self.embedding(x)
        x = self.activation(x)

        x, _ = self.lstm_1(x)

        if self.have_lstm_2:
            x = self.dropout(x)
            x = self.activation(x)
            x, _ = self.lstm_2(x)
        
        logits = self.fc(x)

        if self.do_morphy:
            logits = logits.view(-1, sequence_length, self.num_classes, self.num_c_possibility)

        return logits
    
    def get_number_parameters(self) -> int:
        """ return the number of parameters of the model """
        return sum([prod(param.size()) for param in self.parameters()])
    




if __name__ == '__main__':
    import torch
    from icecream import ic

    model = LSTMClassifier(num_words=10,
                           embedding_size=32,
                           lstm_hidd_size_1=64,
                           lstm_hidd_size_2=32,
                           fc_hidd_size=[64, 128],
                           num_classes=28,
                           num_c_possibility=10)
    
    ic(model)
    ic(model.get_number_parameters())

    x = torch.randint(0, 10, (64, 20))
    ic(x.shape)

    y = model.forward(x)
    ic(y.shape)

