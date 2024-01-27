import os
import torch
import numpy as np
from tqdm import tqdm
from easydict import EasyDict
from icecream import ic

from src.dataloader.dataloader import create_dataloader
from src.model.get_model import get_model
from config.config import test_logger
from src.metrics import get_metrics


def test(config: EasyDict, logging_path: str) -> None:

    # Use gpu or cpu
    if torch.cuda.is_available() and config.learning.device == 'cuda':
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    ic(device)

    # Get data
    test_generator, _ = create_dataloader(config=config, mode='test')
    n_test = len(test_generator)
    ic(n_test)

    # Get model
    model = get_model(config)
    model = model.to(device)
    checkpoint_path = os.path.join(logging_path, 'checkpoint.pt')
    assert os.path.isfile(checkpoint_path), f'Error: model weight was not found in {checkpoint_path}'
    # checkpoint = torch.load(checkpoint_path, map_location=device)
    # ic(checkpoint.keys())
    # exit()
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    ic(model)
    ic(model.get_number_parameters())
    
    # Loss
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')

    # Metrics
    metrics = get_metrics(config=config, device=device)
    metrics_name = metrics.get_metrics_name()
    ic(metrics_name)

    ###############################################################
    # Start Evaluation                                            #
    ###############################################################
    test_loss = 0
    test_range = tqdm(test_generator)
    test_metrics = np.zeros((len(metrics_name)))

    model.eval()
    with torch.no_grad():
        
        for i, (x, y_true) in enumerate(test_range):
            x = x.to(device)
            y_true = y_true.to(device)

            y_pred = model.forward(x)

            if config.task.task_name == 'get_pos':
                    loss = criterion(y_pred.permute(0, 2, 1), y_true)

            else:
                loss = 0
                for c in range(config.task.get_morphy_info.num_classes):
                    loss += criterion(y_pred[:, :, c, :], y_true[:, :, c, :])
                loss = loss / config.task.get_morphy_info.num_classes
            
            test_loss += loss.item()
            test_metrics += metrics.compute(y_true=y_true, y_pred=y_pred)

            current_loss = test_loss / (i + 1)
            test_range.set_description(f"TEST loss: {current_loss:.4f}")
            test_range.refresh()

    ###################################################################
    # Save Scores in logs                                             #
    ###################################################################
    test_loss = test_loss / n_test
    test_metrics = test_metrics / n_test
    
    test_logger(path=logging_path,
                metrics=[config.learning.loss] + metrics_name,
                values=[test_loss] + list(test_metrics))
    
    for i in range(len(metrics_name)):
         print(f'{metrics_name[i]}: {test_metrics[i]}')