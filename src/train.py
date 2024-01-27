import os
import time
import numpy as np
from tqdm import tqdm
from icecream import ic
from easydict import EasyDict

import torch
from torch.optim.lr_scheduler import MultiStepLR

from src.dataloader.dataloader import create_dataloader
from src.model.get_model import get_model
from config.config import train_logger, train_step_logger
from utils.plot_learning_curves import save_learning_curves
from src.metrics import get_metrics


def train(config: EasyDict) -> None:

    # Use gpu or cpu
    if torch.cuda.is_available() and config.learning.device:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    ic(device)

    # Get data
    train_generator, _ = create_dataloader(config=config, mode='train')
    val_generator, _ = create_dataloader(config=config, mode='val')
    n_train, n_val = len(train_generator), len(val_generator)
    ic(n_train, n_val)

    # Get model
    model = get_model(config)
    model = model.to(device)
    ic(model)
    ic(model.get_number_parameters())
    ic(model.state_dict().keys())
    
    # Loss
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')

    # Optimizer and Scheduler
    assert config.learning.optimizer == 'adam', NotImplementedError(
        f"The optimizer '{config.learning.optimizer}' was not implemented. Only 'adam' is inplemented")
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning.learning_rate)
    scheduler = MultiStepLR(optimizer, milestones=config.learning.milesstone, gamma=config.learning.gamma)

    # Metrics
    metrics = get_metrics(config=config, device=device)
    metrics_name = metrics.get_metrics_name()
    ic(metrics_name)

    save_experiment = config.save_experiment
    ic(save_experiment)
    if save_experiment:
        logging_path = train_logger(config, metrics_name=metrics_name)
        best_val_loss = 10e6

    ###############################################################
    # Start Training                                              #
    ###############################################################
    start_time = time.time()

    for epoch in range(1, config.learning.epochs + 1):
        ic(epoch)
        train_loss = 0
        train_metrics = np.zeros((len(metrics_name)))
        train_range = tqdm(train_generator)

        # Training
        model.train()
        for i, (x, y_true) in enumerate(train_range):
            x = x.to(device)
            y_true = y_true.to(device)

            y_pred = model.forward(x)

            if config.task.task_name == 'get_pos':
               loss = criterion(y_pred.permute(0, 2, 1), y_true)

            else:
                loss = 0
                for c in range(config.task.get_morphy_info.num_classes):
                    loss += criterion(y_pred[:, :, c, :], y_true[:, :, c])
                loss = loss / config.task.get_morphy_info.num_classes

            train_loss += loss.item()
            train_metrics += metrics.compute(y_true=y_true, y_pred=y_pred)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            current_loss = train_loss / (i + 1)
            train_range.set_description(f"TRAIN -> epoch: {epoch} || loss: {current_loss:.4f}")
            train_range.refresh()

        ###############################################################
        # Start Validation                                            #
        ###############################################################

        val_loss = 0
        val_metrics = np.zeros((len(metrics_name)))
        val_range = tqdm(val_generator)

        model.eval()

        with torch.no_grad():

            val_loss = 0
            val_metrics = 0
            
            for i, (x, y_true) in enumerate(val_range):
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
                
                val_loss += loss.item()
                val_metrics += metrics.compute(y_true=y_true, y_pred=y_pred)

                current_loss = val_loss / (i + 1)
                val_range.set_description(f"VAL   -> epoch: {epoch} || loss: {current_loss:.4f}")
                val_range.refresh()
        
        scheduler.step()       

        ###################################################################
        # Save Scores in logs                                             #
        ###################################################################
        train_loss = train_loss / n_train
        val_loss = val_loss / n_val
        train_metrics = train_metrics / n_train
        val_metrics = val_metrics / n_val
        
        if save_experiment:
            train_step_logger(path=logging_path, 
                              epoch=epoch, 
                              train_loss=train_loss, 
                              val_loss=val_loss,
                              train_metrics=train_metrics,
                              val_metrics=val_metrics)
            
            if config.learning.save_checkpoint and val_loss < best_val_loss:
                print('save model weights')
                torch.save(model.state_dict(), os.path.join(logging_path, 'checkpoint.pt'))
                best_val_loss = val_loss
                ic(best_val_loss)

    stop_time = time.time()
    print(f"training time: {stop_time - start_time}secondes for {config.learning.epochs} epochs")
    
    if save_experiment:
        save_learning_curves(path=logging_path)

