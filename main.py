import os
import yaml
import argparse
from icecream import ic
from typing import Optional
from easydict import EasyDict

from src.train import train
from src.test import test
from src.infer import infer
from config.process_config import process_config
from src.baseline import baseline_benchmark, baseline_infer


def load_config(path: Optional[str]='config/config.yaml') -> EasyDict:
    stream = open(path, 'r')
    return EasyDict(yaml.safe_load(stream))


def find_config(experiment_path: str) -> str:
    yaml_in_path = list(filter(lambda x: x[-5:] == '.yaml', os.listdir(experiment_path)))

    if len(yaml_in_path) == 1:
        return os.path.join(experiment_path, yaml_in_path[0])

    if len(yaml_in_path) == 0:
        print("ERROR: config.yaml wasn't found in", experiment_path)
    
    if len(yaml_in_path) > 0:
        print("ERROR: a lot a .yaml was found in", experiment_path)
    
    exit()

IMPLEMENTED = ['train', 'baseline', 'test', 'infer']

def main(options: dict) -> None:

    assert options['mode'] in IMPLEMENTED, f"Error, expected mode must in {IMPLEMENTED} but found {options['mode']}"

    if options['mode'] == 'train':
        config = load_config(options['config_path'])
        if options['task'] is not None:
            config.task.task_name = options['task']
        process_config(config)
        ic(config)
        train(config)
    
    if options['mode'] == 'baseline':
        baseline_benchmark.test_dictionary()
    
    if options['mode'] == 'test':
        assert options['path'] is not None, 'Error, please enter the path of your experimentation that you want to test'
        config_path = find_config(experiment_path=options['path'])
        config = load_config(config_path)
        test(config=config, logging_path=options['path'])
    
    if options['mode'] == 'infer':
        assert options['path'] is not None, 'Error, please enter the path of your experimentation that you want to test'
        if options['path'] == 'baseline':
            baseline_infer.run_infer()
        else:
            config_path = find_config(experiment_path=options['path'])
            config = load_config(config_path)
            infer(config=config, logging_path=options['path'], folder='infer')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Options
    parser.add_argument('--mode', '-m', default=None, type=str,
                        help="choose a mode between 'train', 'data'")
    parser.add_argument('--config_path', '-c', default=os.path.join('config', 'config.yaml'),
                        type=str, help="path to config (for training)")
    parser.add_argument('--path', '-p', type=str,
                        help="experiment path (for test, prediction or generate)")
    parser.add_argument('--task', '-t', type=str, default=None,
                        help="task for model (will overwrite the config) for trainning")

    args = parser.parse_args()
    options = vars(args)

    main(options)