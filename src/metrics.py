import torch
import numpy as np
from typing import List, Any
from icecream import ic
from easydict import EasyDict
from torchmetrics import Accuracy


class Metrics:
    def __init__(self, config: EasyDict) -> None:
        if 'metrics' in config.keys():
            self.metrics_name = list(filter(lambda x: config.metrics[x], config.metrics))
        else:
            self.metrics_name = []

        task = config.task.task_name
        self.num_classes = config.task[f'{task}_info'].num_classes
        
        self.metric = {}

    
    def compute(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> np.ndarray:
        raise NotImplementedError
    
    def get_metrics_name(self) -> List[str]:
        raise NotImplementedError


class POS_Metrics(Metrics):
    def __init__(self, config: EasyDict, device: torch.device=None) -> None:
        super().__init__(config)

        self.metric : dict[str, Any] = {
            'acc micro': Accuracy(num_classes=self.num_classes, average='micro', task='multiclass'),
            'acc macro': Accuracy(num_classes=self.num_classes, average='macro', task='multiclass')
        }
        if device is not None:
            for key, value in self.metric.items():
                self.metric[key] = value.to(device)
    
    def compute(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> np.ndarray:
        """ compute metrics
        y_true: torch tensor (index) with a shape of (B, K)
        y_pred: torch.tensor (one hot) with a shape of (B, K, C)
        On ne mets pas de softmax car argmax = argmax(softmax), 
            et l'on regarde que l'argmax car on fait que l'accuracy
        """
        y_pred_argmax = torch.argmax(y_pred, dim=2)
        metrics_value = []

        if 'acc' in self.metrics_name:
            micro = self.metric['acc micro'](y_pred_argmax, y_true)
            metrics_value.append(micro.item())
        
            macro = self.metric['acc macro'](y_pred_argmax, y_true)
            metrics_value.append(macro.item())
        
        return np.array(metrics_value)

    def get_metrics_name(self) -> List[str]:
        metrics_name = []
        if 'acc' in self.metrics_name:
            metrics_name += ['acc micro', 'acc macro']
        return metrics_name


class MOR_Metrics(Metrics):
    def __init__(self, config: EasyDict) -> None:
        super().__init__(config)

        self.sequence_length = config.data.sequence_length
        num_classes = config.task.get_morphy_info.num_classes

        self.elt = self.sequence_length * num_classes
    
    def compute(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> np.ndarray:
        """ compute metrics
        y_true: torch tensor (one hot) with a shape of (B, K, C, N)
        y_pred: torch.tensor (one hot) with a shape of (B, K, C, N)
        """
        y_pred_argmax = torch.argmax(y_pred, dim=-1)
        y_true_argmax = torch.argmax(y_true, dim=-1)
        metrics_value = []

        eg = (y_pred_argmax == y_true_argmax)

        if 'acc' in self.metrics_name:
            acc = eg.sum().item() / (y_true.shape[0] * self.elt)
            metrics_value.append(acc)
        
        if 'allgood' in self.metrics_name:
            eg_sum = torch.sum(eg, dim=-1)
            all_good = torch.sum(eg_sum == self.num_classes).item() / (y_true.shape[0] * self.sequence_length)
            metrics_value.append(all_good)
        
        return np.array(metrics_value)
    
    def get_metrics_name(self) -> List[str]:
        metrics_name = []
        if 'acc' in self.metrics_name:
            metrics_name += ['acc micro']
        if 'allgood' in self.metrics_name:
            metrics_name += ['allgood']
        
        return metrics_name


def get_metrics(config: EasyDict, device: torch.device) -> Metrics:
    task = config.task.task_name
    if task == 'get_pos':
        print('POS Metrics')
        return POS_Metrics(config=config, device=device)

    else:
        print('MOR Metrics')
        return MOR_Metrics(config=config)



if __name__ == "__main__":
    import yaml
    config = EasyDict(yaml.safe_load(open('config/config.yaml', 'r')))

    B = 256     # batch size
    K = 10      # sequence length
    V = 3000    # vocab size
    C_pos = 19  # num classes for pos
    C_mor = 28  # num classes for morphy
    N_mor = 13  # num features for morphy

    # mode: get_pos
    x = torch.randint(0, V, (B, K))
    y_pred = torch.rand((B, K, C_pos))
    y_true = torch.randint(0, C_pos, (B, K))

    ic(x.shape, x.dtype)
    ic(y_pred.shape, y_pred.dtype)
    ic(y_true.shape, y_true.dtype)
    

    metrics = POS_Metrics(config=config)
    ic(metrics.get_metrics_name())
    metrics_value = metrics.compute(y_pred=y_pred, y_true=y_true)
    ic(metrics_value)

    # mode: get_morphy
    x = torch.randint(0, V, (B, K))
    y_pred = torch.rand((B, K, C_mor, N_mor))
    y_true = torch.randint(0, N_mor, (B, K, C_mor))
    y_true = torch.nn.functional.one_hot(y_true, num_classes=N_mor).to(torch.float32)

    ic(x.shape, x.dtype)
    ic(y_pred.shape, y_pred.dtype)
    ic(y_true.shape, y_true.dtype)

    metrics = MOR_Metrics(config=config)
    ic(metrics.get_metrics_name())
    metrics_value = metrics.compute(y_pred=y_pred, y_true=y_true)
    ic(metrics_value)




