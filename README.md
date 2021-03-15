# Project Title

## Table of Contents

- [About](#about)
- [Results](#results)

## About <a name = "about"></a>

Given a CSV file containing a dataset of continuous information with a binary class label, this program will train off of the given dataset and predict the accuracy on the same dataset.

There is a total of 4 synthetic training datasets and a pokemon dataset. The synthetic training datasets do not have any feature labels assigned to them, but the pokemon labels are as follows: 
```
Total,HP,Attack,Defense,Sp. Atk,Sp. Def,Speed,Generation,Type 1_Bug,Type 1_Dark,Type 1_Dragon,Type 1_Electric,Type 1_Fairy,Type 1_Fighting,Type 1_Fire,Type 1_Flying,Type 1_Ghost,Type 1_Grass,Type 1_Ground,Type 1_Ice,Type 1_Normal,Type 1_Poison,Type 1_Psychic,Type 1_Rock,Type 1_Steel,Type 1_Water,Type 2_Bug,Type 2_Dark,Type 2_Dragon,Type 2_Electric,Type 2_Fairy,Type 2_Fighting,Type 2_Fire,Type 2_Flying,Type 2_Ghost,Type 2_Grass,Type 2_Ground,Type 2_Ice,Type 2_Normal,Type 2_Poison,Type 2_Psychic,Type 2_Rock,Type 2_Steel,Type 2_Water, Legendary
```
## Results <a name = "results"></a>
The results for the prediction on each dataset with 5 bins:
```
Predicted with an accuracy of 100.0% for synthetic-1 dataset.
Predicted with an accuracy of 94.5% for synthetic-2 dataset.
Predicted with an accuracy of 87.0% for synthetic-3 dataset.
Predicted with an accuracy of 96.0% for synthetic-4 dataset.
Predicted with an accuracy of 95.44% for pokemon dataset.
```