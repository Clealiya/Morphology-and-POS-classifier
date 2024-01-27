Get `morphy` from sentences using LSTM

<p align="center"><img src=report\morphy.png><p>
Example of morphy.

# Table of Contents
- [Table of Contents](#table-of-contents)
- [Data](#data)
- [Requirements](#requirements)
- [Launch the code](#launch-the-code)
  - [Mode](#mode)
  - [Example](#example)
- [Model](#model)
  - [POS prediction](#pos-prediction)
    - [GET\_POS](#get_pos)
  - [MORPHY prediction](#morphy-prediction)
    - [SUPERTAG](#supertag)
    - [SEPARATE](#separate)
    - [FUSION](#fusion)
- [Results](#results)


# Data

- find the data here: http://hdl.handle.net/11234/1-5287
- Download ud-treebanks-v2.13.tgz
- use `tar -xvzf ud-treebanks-v2.13.tgz` to unstack the tgz file
- and move data in order have a path like:

   .                     
    ├── ...                                     
    ├── data                                   
    │   ├── UD_Abaza-ATD                              
	│   │   ├── abd_atb-ud-test.conllu                     
	│   │   ├── abd_atb-ud-test.txt        
	│   │   └── ...                        
    │   ├── UD_Afrikaans-AfriBooms                               
	│   │   ├── af_afribooms-ud-dev.conllu                      
	│   │   ├── af_afribooms-ud-dev.txt                         
	│   │   ├── af_afribooms-ud-test.conllu                      
	│   │   ├── af_afribooms-ud-test.txt                                 
	│   │   ├── af_afribooms-ud-train.conllu                     
	│   │   ├── af_afribooms-ud-train.txt             
	│   │   └── ...                                             
    │   ├── ...                      
    │   └── ...                                  

# Requirements
To run the code you need python (We use python 3.9.13) and packages that is indicate in [`requirements.txt`](requirements.txt).
You can run the following code to install all packages in the correct versions:
```sh
pip install -r requirements.txt
```

# Launch the code

The [`main.py`](main.py) script is the main entry point for this project. It accepts several command-line arguments to control its behavior:

- `--mode` or `-m`: This option allows you to choose a mode between 'train', 'baseline', 'test', and 'infer'.
- `--config_path` or `-c`: This option allows you to specify the path to the configuration file for training. The default is [`config/config.yaml`](config/config.yaml).
- `--path` or `-p`: This option allows you to specify the experiment path for testing, prediction, or generation.
- `--task` or `-t`: This option allows you to specify the task for the model. This will overwrite the task specified in the configuration file for training.

## Mode
Here's what each mode does:

- [`train`](src/train.py): Trains a model using the configuration specified in the `--config_path` and the task specified in `--task`.
- `baseline`: Runs the baseline benchmark test.
- [`test`](src/test.py): Tests the model specified in the `--path`. You must specify a path.
- [`infer`](src/infer): Runs inference using the model specified in the `--path` It will run inference on some exemple sentences located in "infer/infer.txt" and put the results in "infer/configname_infer.txt". If the path is 'baseline', it will run the baseline inference. You must specify a path.

## Example
Here's an example of how to use the script to train a model:

```sh
python main.py --mode train --config_path config/config.yaml --task get_pos
```
This command will train a model using the configuration specified in [`config/config.yaml`](config/config.yaml) with a `task=get_pos`.


Here's an example of how to run a test on the experiment separete:

```sh
python main.py --mode test --path logs/separete
```

Here is an exemple of how to run inference using the baseline model:
```sh
python main.py --mode infer --path baseline
```

# Model

## POS prediction
input shape: $B \times K$
output shape: $B \times K \times 19$
where $B$ is batch size, $K$ sequence size and $19$ the number of POS classes

### GET_POS
<p align="center"><img src=report\get_pos.png><p>

## MORPHY prediction
input shape: $B \times K$
output shape: $B \times K \times 28 \times 13$
where $B$ is batch size, $K$ sequence size and $28$ the number of MORPHY classes and $13$ the maximun of number possibilities of one morphy.

### SUPERTAG
<p align="center"><img src=report\get_morphy_supertag.png><p>

### SEPARATE
<p align="center"><img src=report\get_morphy_separate.png><p>

### FUSION
<p align="center"><img src=report\get_morphy_fusion.png><p>

# Results

| model         | crossentropy | accuracy micro  | accuracy macro  |
| ------------- | ------------ | --------------- | --------------- |
| *GET_POS*     | 0.204        | 0.944           | 0.816           |

*Table 1: Test results for pos prediction*


| model         | crossentropy | accuracy micro | all good |
| ------------- | ------------ | --------------- | -------- |
| *BASELINE*    | -            | 0.980           | 0.791    |
| *SUPERTAG*    | 1.700        | 0.436           | 0.002    |
| *SEPARATE*    | 1.70         | 0.893           | 0.046    |
| *FUSION*      | 1.698        | 0.884           | 0.154    |

*Table 2: Test results for morphy prediction*



