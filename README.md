# npi Toy
A Toy implementation of neural programmer interpreter with bucket sort.

code structure

Simple training.:
The main funtion in the training.py performs a training with examples

## environment for the bucket sort task
The interactions with the environment(the buckets and the element list) are within environment.py

## npi model all in one
npiCore.py is the whole npi model. Since There is only one target, it is suitable. I'm going to add bubble sort task to this project. So this part will be rewrite to accept the task encoder for state embedding and program embedding. Currently I use manual one hot encoding for the bucket sort task

## Trace creator
The trace.py is used to create an excecuting trace for data generation.

## Data generator
The generateData.py can generate bucket sort data for training.
