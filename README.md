# reslogit-example

This project shows an example of a reslogit model using a Theano backend DL library.

## File description

```main.py``` script to run program

```models.py``` class objects for ResLogit, MLP, etc.

```utils.py``` misc. tools to visualize data

```core.py``` extra core functions

```optimizers/``` different optimizers for gradient descent 
- sgd
- sgd momentum
- rmsprop

```data-20190702_2.csv``` sample dataset for model estimation and validation

```config.yaml``` configuration file for hyperparameters

```curves.py``` script to generate plots for validation results

## Dataset

This project uses a sample of the Mtl Trajet dataset

### Discrete variable numbering values

**Mode**

1. Auto
2. Vélo
3. Transport Collectif
4. À pied
5. Auto + TC
6. Autre mode
7. Autre combinasion
8. N/A

**Activity**

1. Éducation
2. Santé 
3. Loisir
4. Repas / collation / café
5. Déposer / Ramasser
6. Magasinage / Commision
7. Retour à la maison
8. Travail
9. Réunion pour le travail
10. N/A

## Getting started

```main.py``` provides the script to run to reproduce the results of the ResLogit model estimation. 
```config.yaml``` is the configuration file fir different hyperparameter settings.

### Prerequisites

Python 3.5+ (with pip3), Numpy, Scipy, Pandas, Theano

Matplotlib is used for visualization


### Installation

A ```requirements.txt``` file is provided to install the required library packages through pip

- clone or download the git project repository, and in the project folder run the following to install the reuqirements

Ubuntu (Unix)

The following system packages are required to be installed

```
apt-get install python3 python3-dev pip3
python3 --version
>>> Python 3.X.X
```

Install requirements with pip with --user option

```
cd project-root-folder/
pip3 install --user -r requirements.txt
```

The above command also installs the latest Theano from github.com/Theano/Theano

## Versioning

0.1 inital version 

## Authors

Melvin Wong ([Github](https://github.com/mwong009))

## Licence

This project is licensed under the MIT - see [LICENSE](https://github.com/LiTrans/reslogit-example/blob/master/LICENSE.md) for details
