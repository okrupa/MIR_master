MIR_master
==============================
Praca magisterka - "Wykorzystanie uczenia semi-nadzorowanego w sieciach prototypowych dla zadania klasyfikacji muzyki"
Master thesis "Application of semi-supervised learning in prototypical networks for music classification task"

# Prototypical Network prediction APP

This library was created to allow the user to train his own semi-supervised prototype network and on its basis to tag audio files.

To prepare the environment, run commands from the **setup.sh** file.

To train your model, go to the *src* folder and then train choosen prototypical network
```
cd src
train.py --config_path config.yaml --model_output_path .\trained_models
```

To provide labels for audio files, go to the *src* folder and call the following code
```
cd src
predict.py --config_path config.yaml --model_path {.\trained_models\choosen_model.ckpt}
e.g. predict.py --config_path config.yaml --model_path .\trained_models\MUSIC.ckpt
```

All cofiguration variables are located in the ***config.yaml*** file. 
You can use 5 approaches to train your network: softkmeans, softkmeansdistractor, softkmeansmasked, music, musicdistractor. For each approach, the user has the option to train with a class of distractors, the number and quantity of which should be sat in the configuration file.

There are 3 datasets to choose from: tiny_sol, gtzan, custom.

The custom dataset should have the following structure 
```
├── dataset_path

    ├── labeled
    
    │   ├── first_class_name
    
    │   │   ├── audio files ...
    
    │   ├── second_class_name
    
    │   │   ├── audio files ..
    
    │   │   ...
    
    ├── unlabeled
    
    │   ├── audio files ...
```    

The folder with the files that the user wants to predict should be set in the *predict_path* variable, along with the specified classes set in *classes* variable. The model can be additionally trained for the given data and a selected set, including your own, by setting the number of episodes using *n_train_dataset*.
