# cNODE-pytorch

This repository is an unofficial pytorch implementation paper : "Predicting microbiome compositions from species assemblages through deep learning" 
(doi: https://doi.org/10.1101/2021.06.17.448886).

The original julia code for this paper is at [here](https://github.com/michel-mata/cNODE-paper).

## Features

Implement the classic cNODE structure.


## Pre-requisite 

1. Make sure you clone the origianl julia repository [here](https://github.com/michel-mata/cNODE-paper).
2. Install pytorch.
3. Install [torchdiffeq](https://github.com/rtqichen/torchdiffeq). 

## Run an experiment 

We current support experiments: `["Drosophila_Gut", "Soil_Vitro", "Human_Gut", "Soil_Vivo"]`.

To run one experiment, for example `Drosophila_Gut`, do 

```shell
python train.py --name Human_Gut --root <path-to-original-julia-repository>
```
where `--root` specifies the path to the original julia code with `Data` folder in it. 

The log files will be output to `./log` directory, and The

## Result

The average result in the julia repository is 0.1164. 

Our results are 0.1393.
