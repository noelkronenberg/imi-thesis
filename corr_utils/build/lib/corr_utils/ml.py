# author: Noel Kronenberg
# version: 01.09.2024 16.00 [WIP]

# imports 

import pandas as pd
import numpy as np
from enum import Enum

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from imblearn.over_sampling import RandomOverSampler # imbalanced-learn

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# custom ML (DL)

# references:
# - https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html
# - https://github.com/flatplanet/Pytorch-Tutorial-Youtube/blob/main/simple_NeuralNetwork.ipynb

# data

def preprocessing(df:pd.DataFrame, variables:list, outcome:str, scale:bool=True, resample:bool=False, tensor:bool=True, test_size:float=0.2) -> 'tuple[torch.FloatTensor, torch.FloatTensor, torch.LongTensor, torch.LongTensor]':
    """
    Preprocesses a dataset by scaling, resampling, and converting to tensors.

    Parameters:
        df (pd.DataFrame): The input dataframe.
        variables (list): The list of feature column names.
        outcome (str): The target column name.
        scale (bool, optional): Whether to scale the features (range(0,1)). Defaults to True.
        resample (bool, optional): Whether to perform (random) oversampling on the dataset. Defaults to False.
        tensor (bool, optional): Whether to convert the data into PyTorch tensors. Defaults to True.
        test_size (float, optional): The proportion of the dataset to include in the test split. Defaults to 0.2.

    Returns:
        tuple[torch.FloatTensor, torch.FloatTensor, torch.LongTensor, torch.LongTensor]: A tuple containing the train and test sets as tensors or arrays (X_train, X_test, y_train, y_test).
    """

    df_patients_ML = df.copy()

    # extract data
    df_X = df_patients_ML[variables]
    df_y = df_patients_ML[outcome]

    # over sampling (reference: https://github.com/dataprofessor/imbalanced-data/blob/main/imbalanced_learn.ipynb)
    if resample:
        ros = RandomOverSampler(sampling_strategy=1)
        df_X, df_y = ros.fit_resample(df_X, df_y)
        print(f'Value counts: \n{df_y.value_counts()}')

    # convert to array
    X = df_X.values
    y = df_y.values

    # scale data
    if scale:
        scaler = MinMaxScaler(feature_range=(0, 1))
        X = scaler.fit_transform(X)

    # split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    if tensor:
        # convert to tensors
        X_train = torch.FloatTensor(X_train)
        X_test = torch.FloatTensor(X_test)

        y_train = torch.LongTensor(y_train)
        y_test = torch.LongTensor(y_test)

    return X_train, X_test, y_train, y_test

def get_eda_metrics(X_train:torch.Tensor, X_test:torch.Tensor, y_train:torch.Tensor, y_test:torch.Tensor, variables:list, outcome:str) -> None:
    """
    Generates exploratory data analysis (EDA) metrics and plots.

    Parameters:
        X_train (torch.Tensor): Training features.
        X_test (torch.Tensor): Test features.
        y_train (torch.Tensor): Training labels.
        y_test (torch.Tensor): Test labels.
        variables (list): List of feature names.
        outcome (str): Name of the target variable.

    Returns:
        None: Displays histograms of features and outcome.
    """
       
    X_train_np = X_train.numpy()
    X_test_np = X_test.numpy()
    y_train_np = y_train.numpy()
    y_test_np = y_test.numpy()

    X_combined_np = np.vstack((X_train_np, X_test_np))
    y_combined_np = np.concatenate((y_train_np, y_test_np))
    df_combined = pd.DataFrame(X_combined_np, columns=variables)
    df_combined[outcome] = y_combined_np

    num_vars = len(variables)
    num_plots = num_vars + 1
    num_rows = (num_plots + 2) // 3
    fig, axes = plt.subplots(nrows=num_rows, ncols=3, figsize=(10, num_rows * 3))
    axes = axes.flatten()

    for i, var in enumerate(variables + [outcome]):
        ax = axes[i]
        df_combined[var].plot(kind='hist', bins=30, ax=ax, title=var)
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def get_class_weights(y_train:torch.Tensor, show:bool=False):
    """
    Calculates class weights for imbalanced datasets.

    Parameters:
        y_train (torch.Tensor): Training labels.
        show (bool, optional): Whether to print the class weights. Defaults to False.

    Returns:
        torch.Tensor: Tensor containing class weights.
    """

    class_counts = torch.bincount(y_train)
    total_samples = len(y_train)

    # class weights as the inverse of class frequency
    class_weights = total_samples / (len(class_counts) * class_counts.float())

    if show:
        print(f'Class weights: {class_weights}')

    return class_weights

# model

class Model(Enum):
    """
    Enumeration for selecting different model architectures.
    """

    CUSTOM = 1
    RESNET = 2 # vision

class NeuralNetwork(nn.Module):
    """
    Custom fully connected neural network.

    Parameters:
        in_features (int): Number of input features.
        h1 (int, optional): Number of neurons in the first hidden layer. Defaults to 16.
        h2 (int, optional): Number of neurons in the second hidden layer. Defaults to 16.
        h3 (int, optional): Number of neurons in the third hidden layer. Defaults to 16.
        out_features (int, optional): Number of output features. Defaults to 2.
    """

    def __init__(self, in_features, h1=16, h2=16, h3=16, out_features=2): # specify width
        super().__init__() # instantiate nn.Module
        self.fc1 = nn.Linear(in_features, h1) # fully connected first hidden layer
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, h3)
        self.out = nn.Linear(h3, out_features) # output layer

    def forward(self, x):
        x = F.relu(self.fc1(x)) # start with first hidden layer and ReLu
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.out(x) # output layer
        # x = F.softmax(x, dim=1) # get probabilities
        return x
    
class ResNet(nn.Module):
    """
    ResNet-based neural network model.

    Parameters:
        num_classes (int, optional): Number of output classes. Defaults to 2.
    """

    def __init__(self, num_classes=2):
        super(ResNet, self).__init__()
        # pre-trained ResNet
        self.model = models.resnet18(pretrained=True)
        # replace final fully connected layer
        self.model.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.model(x)

def get_model(features:int, class_weights_tensor:torch.Tensor=None, selected_model:Model=Model.CUSTOM) -> tuple:
    """
    Instantiates and returns a model, criterion, and optimizer.

    Parameters:
        features (int): Number of input features.
        class_weights_tensor (torch.Tensor, optional): Tensor containing class weights. Defaults to None.
        selected_model (Model, optional): Enum to select model architecture. Defaults to Model.CUSTOM.

    Returns:
        tuple: A tuple containing the model, criterion, and optimizer.
    """
     
    # create a model instance
    if selected_model == Model.CUSTOM:
        model = NeuralNetwork(features)
    elif selected_model == Model.RESNET:
        model = ResNet(features)

    # criterion of model to measure the error
    if class_weights_tensor != None:
        # with class weights
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    else:
        # without class weights
        criterion = nn.CrossEntropyLoss()

    # optimizer and learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    return model, criterion, optimizer

# training

def train(X_train:torch.Tensor, y_train:torch.Tensor, model:NeuralNetwork, criterion, optimizer, epochs:int=100) -> None:
    """
    Trains a neural network model.

    Parameters:
        X_train (torch.Tensor): Training features.
        y_train (torch.Tensor): Training labels.
        model (NeuralNetwork): The neural network model.
        criterion: Loss function.
        optimizer: Optimizer for gradient descent.
        epochs (int, optional): Number of training epochs. Defaults to 100.

    Returns:
        None: Displays the training loss over epochs.
    """

    losses = [] 
    model.train()

    for i in range(epochs):
        y_pred = model(X_train) # get predictions
        
        # save loss
        loss = criterion(y_pred, y_train)
        losses.append(loss.detach().numpy()) # turn into number

        if i % 10 == 0:
            print(f'Epoch: {i} and loss: {loss}')

        # back propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    plt.plot(range(epochs), losses)
    plt.ylabel('loss')
    plt.xlabel('Epoch')
    plt.show()

# evaluation

def evaluate(model:NeuralNetwork, X_train:torch.Tensor, y_train:torch.Tensor, X_test:torch.Tensor, y_test:torch.Tensor) -> None:
    """
    Evaluates a model on the training and test sets.

    Parameters:
        model (NeuralNetwork): The trained neural network model.
        X_train (torch.Tensor): Training features.
        y_train (torch.Tensor): Training labels.
        X_test (torch.Tensor): Testing features.
        y_test (torch.Tensor): Testing labels.

    Returns:
        None: Prints the results.
    """

    print('Performance on training set: \n')
    model.eval()
    with torch.no_grad():
        y_pred = model(X_train)
        y_pred = torch.argmax(F.softmax(y_pred, dim=1), dim=1)
        print(classification_report(y_train, y_pred))

    print('Performance on test set: \n')
    with torch.no_grad():
        y_pred = model(X_test)
        y_pred = torch.argmax(F.softmax(y_pred, dim=1), dim=1)
        print(classification_report(y_test, y_pred)) 

    print('AU-ROC on test set: ')
    with torch.no_grad():
        y_pred = model(X_test)[:,1]
        auc_score = roc_auc_score(y_test, y_pred)
        print(auc_score) 

    print('\n')

    print('AU-PRC on test set: ')
    with torch.no_grad():
        y_pred = model(X_test)[:,1]
        auc_score = average_precision_score(y_test, y_pred)
        print(auc_score, f' (baseline: {sum(y_test) / len(y_test)})') 

# prediction

def get_prediction(data:torch.Tensor, model:NeuralNetwork) -> None: # e.g. torch.FloatTensor([1, 1, 1, 1, 1])
    """
    Predicts the class and probabilities for the given input data.

    Parameters:
        data (torch.Tensor): Input tensor for prediction.
        model (NeuralNetwork): The neural network model.

    Returns:
        None: Prints the prediction.
    """

    model.eval()

    if data.dim() == 1:
        data = data.unsqueeze(0)

    with torch.no_grad():
        y_pred = model(data)
        y_pred_proba = F.softmax(y_pred, dim=1)
        predicted_class = y_pred.argmax(dim=1).item()
        prob_class_0 = y_pred_proba[0, 0].item()
        prob_class_1 = y_pred_proba[0, 1].item()

    print(f'Predicted class: {predicted_class}')
    print(f'Probability for class 0: {prob_class_0:.2f}')
    print(f'Probability for class 1: {prob_class_1:.2f}')

# weights

def get_weights(model:NeuralNetwork, variables:list) -> pd.DataFrame:
    """
    Retrieves feature weights of a neural network.

    Parameters:
        model (NeuralNetwork): The neural network model.
        variables (list): List of feature names.

    Returns:
        pd.DataFrame: DataFrame containing feature importance scores.
    """

    weights = model.fc1.weight.data
    feature_importance = torch.abs(weights).sum(dim=0)
    normalized_importance = feature_importance / feature_importance.sum()

    df_feature_importance = pd.DataFrame({'variable': variables, 'weight': normalized_importance.tolist()})
    df_feature_importance['weight'] = df_feature_importance['weight'].round(2)
    
    return df_feature_importance

# export

def export_model_state(model:NeuralNetwork, path:str, name:str) -> None:
    """
    Exports a model's state dictionary to a file as well as the onnx.

    Parameters:
        model (NeuralNetwork): The neural network model.
        path (str): The directory path where the model should be saved.
        name (str): The name of the file to save the model state.

    Returns:
        None
    """

    # reference: https://pytorch.org/tutorials//beginner/onnx/export_simple_model_to_onnx_tutorial.html
    dummy_input = torch.randn(1, model.fc1.in_features)
    torch.onnx.export(model, dummy_input, 
                      f'{path + name}.onnx', 
                      export_params=True,
                      input_names=["input"], 
                      output_names=["output"])
    
    torch.save(model.state_dict(), path + name)

def import_model(path:str, name:str) -> NeuralNetwork:
    """
    Imports a model's state dictionary from a file.

    Parameters:
        path (str): The directory path from where the model should be loaded.
        name (str): The name of the file containing the model state.

    Returns:
        NeuralNetwork: The neural network model with the loaded state.
    """

    new_model = NeuralNetwork()
    new_model.load_state_dict(torch.load(path + name))
    
# wrapper

def wrapper(df:pd.DataFrame, variables:list, outcome:str, epochs=100) -> None:
    """
    A wrapper function to preprocess data, train, and evaluate a model.

    Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        variables (list): List of feature column names.
        outcome (str): The name of the outcome column.
        epochs (int, optional): Number of epochs to train. Defaults to 100.

    Returns:
        None: Prints the results.
    """

    X_train, X_test, y_train, y_test = preprocessing(df, variables, outcome)
    class_weights_tensor = get_class_weights(y_train)
    model, criterion, optimizer = get_model(len(variables), class_weights_tensor)
    train(X_train, y_train, model, criterion, optimizer, epochs)
    evaluate(model, X_train, y_train, X_test, y_test)


# AutoML

from pycaret.classification import * # pip install --upgrade 'joblib<1.4' pycaret (https://github.com/pycaret/pycaret/issues/3959)

# reference: https://pycaret.gitbook.io/docs/get-started/quickstart; https://www.datacamp.com/blog/top-python-libraries-for-data-science
def get_autoML(data:pd.DataFrame, target:str, model_name:str=None, save:bool=False) -> object:
    """
    Performs AutoML using PyCaret (https://github.com/pycaret/pycaret/) to compare models and evaluate the best one.

    Parameter:
        data (pd.DataFrame): The dataframe containing (all) the data.
        target (str): The name of the target column.
        model_name (str, optional): The name to save the best model. Defaults to None.
        save (bool, optional): Whether to save the best model. Defaults to False.

    Returns:
        object: The best model found by PyCaret.
    """
     
    # initialize training environment and create transformation pipeline
    s = ClassificationExperiment()
    s.setup(data=data, target=target, session_id=42)

    # train and evaluate all estimators using cross-validation
    # prints scoring grid with average cross-validated scores
    best = s.compare_models()
    print(best)

    # analyze the performance of a trained model on test set
    s.evaluate_model(best)

    # scores the data and returns prediction_label and prediction_score probability of the predicted class
    # when data is None, predicts on test set 
    # note: Score = probability of the predicted class (not positive)
    s.predict_model(best)

    # show pipeline (reference: https://nbviewer.org/github/pycaret/examples/blob/main/PyCaret%202%20Classification.ipynb)
    from sklearn import set_config
    set_config(display='diagram')
    best
    set_config(display='text')

    if save:
        # save model
        s.save_model(best, model_name)

    return best

def load_autoML(model_name:str) -> object:
    """
    Loads a saved PyCaret model.

    Parameters:
        model_name (str): The name of the model file to load.

    Returns:
        object: The loaded PyCaret model.
    """

    # load model
    s = ClassificationExperiment()
    loaded_model = s.load_model(model_name)

    print(loaded_model)

    return loaded_model