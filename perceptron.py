import numpy as np
import torch
#x is your input data training is the training input
def perceptron(x, training):
    iterations = 200
    alpha = .2
    midnode = 50
    startNode, samples = x.shape
    outputNode = training.shape[0]
    X=torch.zeros((2,samples))
    X = torch.cat(x,torch.zeros(samples))
