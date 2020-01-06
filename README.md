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

## Discrete variable numbering values

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
