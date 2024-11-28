import os, sys

from pathlib import Path
path_root = Path(__file__).parents[4]  # upto 'codebase' folder
sys.path.insert(0, str(path_root))
# print(sys.path)

from utils import dl_reproducible_result_util

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from postproc.mat_p2ip_pd.pul.proc_pul import kan
from utils import PPIPUtils


# Check if GPU is available and set the device accordingly
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
num_mut_pts_feat_wt = 15


def pul_prepare_data(train_train_df=None, train_val_df=None, test_df=None):
    print('inside pul_prepare_data() method - Start')
    # Drop columns 'prot_id', 'chain_1_seq', 'chain_2_seq' from all the dataframes
    col_to_be_dropped = ['prot_id', 'chain_1_seq', 'chain_2_seq']
    train_train_df = train_train_df.drop(col_to_be_dropped, axis=1)
    train_val_df = train_val_df.drop(col_to_be_dropped, axis=1)
    test_df = test_df.drop(col_to_be_dropped, axis=1)

    # label convention: 0 for negative, 1 for positive and -1 for unlabelled
    # Separate labelled and unlabelled data from train_train_df (train_train_df contains 
    # only positive and unlabelled data)
    train_train_positive_data = train_train_df[train_train_df['label'] == 1]
    train_train_positive_data = train_train_positive_data.reset_index(drop=True)
    train_train_unlabelled_data = train_train_df[train_train_df['label'] == -1]
    train_train_unlabelled_data = train_train_unlabelled_data.reset_index(drop=True)

    # Sample from the train_train_unlabelled_data to include in the initial training for finding reliable negatives
    # ## train_train_positive_data_size = train_train_positive_data.shape[0]
    # ## U_sample = train_train_unlabelled_data.sample(n=train_train_positive_data_size, random_state=456)
    U_sample = train_train_unlabelled_data.sample(frac=0.2, random_state=456)
    initial_train_data = pd.concat([train_train_positive_data, U_sample])
    initial_train_data = initial_train_data.reset_index(drop=True)

    # Prepare the training data
    X_train = initial_train_data.drop('label', axis=1).values
    y_train = initial_train_data['label'].replace(-1, 0).values  # Treat sampled unlabelled as negative initially

    X_val = train_val_df.drop('label', axis=1).values
    y_val = train_val_df['label'].values

    X_test = test_df.drop('label', axis=1).values
    y_test = test_df['label'].values

    # Standardize the features
    scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_val = scaler.transform(X_val)
    # X_test = scaler.transform(X_test)

    # ####### Throughout the training and inferencing phases of PUL, more weightage is given to 'num_mut_pts' feature
    #  and normalization is avoided for this feature.
    skip_col_train = X_train[:, 1] * num_mut_pts_feat_wt  # This skips normalization for the 1st column
    # Normalize the rest of the columns (all columns except index 1)
    train_arr_to_normalize = np.delete(X_train, 1, axis=1)  # Remove the 1st column for normalization
    train_arr_normalized = scaler.fit_transform(train_arr_to_normalize)
    # Combine the unnormalized column back with the normalized array
    X_train = np.insert(train_arr_normalized, 1, skip_col_train, axis=1)  # Insert skipped column back at index 1

    skip_col_val = X_val[:, 1] * num_mut_pts_feat_wt  # This skips normalization for the 1st column
    # Normalize the rest of the columns (all columns except index 1)
    val_arr_to_normalize = np.delete(X_val, 1, axis=1)  # Remove the 1st column for normalization
    val_arr_normalized = scaler.fit_transform(val_arr_to_normalize)
    # Combine the unnormalized column back with the normalized array
    X_val = np.insert(val_arr_normalized, 1, skip_col_val, axis=1)  # Insert skipped column back at index 1

    skip_col_test = X_test[:, 1] * num_mut_pts_feat_wt  # This skips normalization for the 1st column
    # Normalize the rest of the columns (all columns except index 1)
    test_arr_to_normalize = np.delete(X_test, 1, axis=1)  # Remove the 1st column for normalization
    test_arr_normalized = scaler.fit_transform(test_arr_to_normalize)
    # Combine the unnormalized column back with the normalized array
    X_test = np.insert(test_arr_normalized, 1, skip_col_test, axis=1)  # Insert skipped column back at index 1

    print('inside pul_prepare_data() method - End')
    return X_train, y_train, X_val, y_val, X_test, y_test, scaler


def calculate_class_weights(labels):
    """
    Calculate class weights for handling class imbalance.

    Args:
        labels (np.array): Array of labels (binary).

    Returns:
        torch.Tensor: A single weight for the positive class, to be used in pos_weight of BCEWithLogitsLoss.
    """
    num_pos = sum(labels)
    num_neg = len(labels) - num_pos
    # pos_weight is the ratio of negative samples to positive samples
    pos_weight = num_neg / num_pos

    class_weights = torch.FloatTensor([pos_weight]).to(device)
    return class_weights


def train_model(X_train, y_train, X_val, y_val, input_size, num_epochs=20, lr=0.001, class_weights=None, hparams={}):
    """
    Train the model and evaluate it on the validation set at each epoch.

    Args:
        X_train (torch.Tensor): Training features.
        y_train (torch.Tensor): Training labels.
        X_val (torch.Tensor): Validation features.
        y_val (torch.Tensor): Validation labels.
        input_size (int): Number of input features.
        num_epochs (int, optional): Number of epochs for training. Defaults to 20.
        lr (float, optional): Learning rate. Defaults to 0.001.
        class_weights (torch.Tensor, optional): Class weights for handling imbalance. Defaults to None.

    Returns:
        Trained model.
    """
    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
    X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(device)

    # Initialize the model, loss function, and optimizer
    grid_size = hparams['grid_size']
    spline_order = hparams['spline_order']
    scale_noise = hparams['scale_noise'] 
    scale_base = hparams['scale_base']
    scale_spline = hparams['scale_spline']

    model = kan.KAN([input_size, 256, 1], grid_size=grid_size, spline_order=spline_order 
                    , scale_noise=scale_noise, scale_base=scale_base, scale_spline=scale_spline).to(device)
    # criterion = nn.BCEWithLogitsLoss()
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights) if class_weights is not None else nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

    # Train the model
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        
        outputs = model(X_train)
        loss = criterion(outputs.squeeze(), y_train.squeeze())
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs.squeeze(), y_val.squeeze())
        # End of with block
        # Update learning rate
        scheduler.step()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Validation Loss: {val_loss.item():.4f}')
    # End of for loop: for epoch in range(num_epochs):
    return model
# End of train_model() method


def train_model_minibatch(X_train, y_train, X_val, y_val, input_size, num_epochs=20, lr=0.001, class_weights=None, hparams={}, batch_size=512):
    """
    Train the model and evaluate it on the validation set at each epoch.

    Args:
        X_train (torch.Tensor): Training features.
        y_train (torch.Tensor): Training labels.
        X_val (torch.Tensor): Validation features.
        y_val (torch.Tensor): Validation labels.
        input_size (int): Number of input features.
        num_epochs (int, optional): Number of epochs for training. Defaults to 20.
        lr (float, optional): Learning rate. Defaults to 0.001.
        class_weights (torch.Tensor, optional): Class weights for handling imbalance. Defaults to None.

    Returns:
        Trained model.
    """
    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
    X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(device)

    # Initialize the model, loss function, and optimizer
    grid_size = hparams['grid_size']
    spline_order = hparams['spline_order']
    scale_noise = hparams['scale_noise'] 
    scale_base = hparams['scale_base']
    scale_spline = hparams['scale_spline']

    model = kan.KAN([input_size, 256, 1], grid_size=grid_size, spline_order=spline_order 
                    , scale_noise=scale_noise, scale_base=scale_base, scale_spline=scale_spline).to(device)
    # criterion = nn.BCEWithLogitsLoss()
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights) if class_weights is not None else nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

    # Create DataLoader for minibatch training
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    # Train the model
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs.squeeze(), y_batch.squeeze())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        # End of for loop: for X_batch, y_batch in train_loader:
        avg_loss = running_loss / len(train_loader)

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs.squeeze(), y_val.squeeze())
        # End of with block
        # Update learning rate
        scheduler.step()
        print(f'Epoch [{epoch+1}/{num_epochs}], avg_loss: {avg_loss:.4f}, Validation Loss: {val_loss.item():.4f}')
    # End of for loop: for epoch in range(num_epochs):
    return model
# End of train_model_minibatch() method


def identify_reliable_negatives(model, unlabelled_data, scaler, threshold=0.1):
    """
    Identify reliable negatives from the unlabelled data.

    Args:
        model: Trained model.
        unlabelled_data (pd.DataFrame): Unlabelled data.
        scaler (StandardScaler): Scaler used for data normalization.
        threshold (float, optional): Threshold to identify reliable negatives. Defaults to 0.1.

    Returns:
        pd.DataFrame: Reliable negatives identified from the unlabelled data.
    """
    # X_unlabelled = scaler.fit_transform(unlabelled_data.drop('label', axis=1).values)

    # ####### Throughout the training and inferencing phases of PUL, more weightage is given to 'num_mut_pts' feature
    #  and normalization is avoided for this feature.
    X_unlabelled = unlabelled_data.drop('label', axis=1).values
    skip_col_unlabelled = X_unlabelled[:, 1] * num_mut_pts_feat_wt  # This skips normalization for the 1st column
    # Normalize the rest of the columns (all columns except index 1)
    unlabelled_arr_to_normalize = np.delete(X_unlabelled, 1, axis=1)  # Remove the 1st column for normalization
    unlabelled_arr_normalized = scaler.fit_transform(unlabelled_arr_to_normalize)
    # Combine the unnormalized column back with the normalized array
    X_unlabelled = np.insert(unlabelled_arr_normalized, 1, skip_col_unlabelled, axis=1)  # Insert skipped column back at index 1

    X_unlabelled = torch.tensor(X_unlabelled, dtype=torch.float32).to(device)

    model.eval()
    with torch.no_grad():
        # unlabelled_pred_proba = model(X_unlabelled).cpu().numpy()
        logits = model(X_unlabelled)
        unlabelled_pred_proba = torch.sigmoid(logits.squeeze())
    # End of with block
    unlabelled_pred_proba = unlabelled_pred_proba.cpu().numpy()
    reliable_negatives = unlabelled_data[unlabelled_pred_proba.flatten() < threshold]
    return reliable_negatives


def retrain_model(model, new_train_data, X_val, y_val, scaler, num_epochs=20, class_weights=None):
    """
    Retrain the model on the new training data with validation.

    Args:
        model: Initial trained  model.
        new_train_data (pd.DataFrame): New training data including reliable negatives.
        X_val (np.array): Validation features.
        y_val (np.array): Validation labels.
        scaler (StandardScaler): Scaler used for data normalization.
        num_epochs (int, optional): Number of epochs for retraining. Defaults to 20.
        class_weights (torch.Tensor, optional): Class weights for handling imbalance. Defaults to None.

    Returns:
        Retrained model.
    """
    X_new_train = new_train_data.drop('label', axis=1).values
    y_new_train = new_train_data['label'].values
    
    # X_new_train = scaler.fit_transform(X_new_train) 

    # ####### Throughout the training and inferencing phases of PUL, more weightage is given to 'num_mut_pts' feature
    #  and normalization is avoided for this feature.
    skip_col_new_train = X_new_train[:, 1] * num_mut_pts_feat_wt  # This skips normalization for the 1st column
    # Normalize the rest of the columns (all columns except index 1)
    new_train_arr_to_normalize = np.delete(X_new_train, 1, axis=1)  # Remove the 1st column for normalization
    new_train_arr_normalized = scaler.fit_transform(new_train_arr_to_normalize)
    # Combine the unnormalized column back with the normalized array
    X_new_train = np.insert(new_train_arr_normalized, 1, skip_col_new_train, axis=1)  # Insert skipped column back at index 1

    X_new_train = torch.tensor(X_new_train, dtype=torch.float32).to(device)
    y_new_train = torch.tensor(y_new_train, dtype=torch.float32).unsqueeze(1).to(device)

    X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(device)
    # ###################################################################
    # ## model = kan.KAN([X_new_train.shape[1], 128, 1]).to(device)  # ??Re-initialize?? 
    # ###################################################################

    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights) if class_weights is not None else nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

    # Retrain the model
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        outputs = model(X_new_train)
        loss = criterion(outputs.squeeze(), y_new_train.squeeze())
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs.squeeze(), y_val.squeeze())
        # End of with block
        # Update learning rate
        scheduler.step()
        print(f'Retrain Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Validation Loss: {val_loss.item():.4f}')
    # End of for loop: for epoch in range(num_epochs):
    return model
# End of retrain_model() method


def retrain_model_minibatch(model, new_train_data, X_val, y_val, scaler, num_epochs=20, class_weights=None, batch_size=512):
    """
    Retrain the model on the new training data with validation.

    Args:
        model: Initial trained  model.
        new_train_data (pd.DataFrame): New training data including reliable negatives.
        X_val (np.array): Validation features.
        y_val (np.array): Validation labels.
        scaler (StandardScaler): Scaler used for data normalization.
        num_epochs (int, optional): Number of epochs for retraining. Defaults to 20.
        class_weights (torch.Tensor, optional): Class weights for handling imbalance. Defaults to None.

    Returns:
        Retrained model.
    """
    X_new_train = new_train_data.drop('label', axis=1).values
    y_new_train = new_train_data['label'].values
    
    # X_new_train = scaler.fit_transform(X_new_train) 

    # ####### Throughout the training and inferencing phases of PUL, more weightage is given to 'num_mut_pts' feature
    #  and normalization is avoided for this feature.
    skip_col_new_train = X_new_train[:, 1] * num_mut_pts_feat_wt  # This skips normalization for the 1st column
    # Normalize the rest of the columns (all columns except index 1)
    new_train_arr_to_normalize = np.delete(X_new_train, 1, axis=1)  # Remove the 1st column for normalization
    new_train_arr_normalized = scaler.fit_transform(new_train_arr_to_normalize)
    # Combine the unnormalized column back with the normalized array
    X_new_train = np.insert(new_train_arr_normalized, 1, skip_col_new_train, axis=1)  # Insert skipped column back at index 1

    X_new_train = torch.tensor(X_new_train, dtype=torch.float32).to(device)
    y_new_train = torch.tensor(y_new_train, dtype=torch.float32).unsqueeze(1).to(device)

    X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(device)
    # ###################################################################
    # ## model = kan.KAN([X_new_train.shape[1], 128, 1]).to(device)  # ??Re-initialize?? 
    # ###################################################################

    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights) if class_weights is not None else nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

    # Create DataLoader for minibatch training
    new_train_dataset = TensorDataset(X_new_train, y_new_train)
    new_train_loader = DataLoader(dataset=new_train_dataset, batch_size=batch_size, shuffle=True)

    # Retrain the model
    for epoch in range(num_epochs):
        model.train()
        new_running_loss = 0.0
        
        for new_X_batch, new_y_batch in new_train_loader:
            optimizer.zero_grad()
            outputs = model(new_X_batch)
            loss = criterion(outputs.squeeze(), new_y_batch.squeeze())
            loss.backward()
            optimizer.step()
            new_running_loss += loss.item()
        # End of for loop: for new_X_batch, new_y_batch in new_train_loader:
        new_avg_loss = new_running_loss / len(new_train_loader)

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs.squeeze(), y_val.squeeze())
        # End of with block
        # Update learning rate
        scheduler.step()
        print(f'Retrain Epoch [{epoch+1}/{num_epochs}], avg_loss: {new_avg_loss:.4f}, Validation Loss: {val_loss.item():.4f}')
    # End of for loop: for epoch in range(num_epochs):
    return model
# End of retrain_model_minibatch() method


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model on labelled test data

    Args:
        model: Trained model to be evaluated.
        X_test: Test data feature set.
        y_test: Test data labels.

    Returns:
        dictionary: Various performance scores of the model on the labelled data.
    """
    model.eval()

    # Prepare the test data
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

    # Perform inference
    with torch.no_grad():
        logits = model(X_test)
        y_pred_proba = torch.sigmoid(logits.squeeze())

    # Assuming y_test is the ground truth labels and y_pred_proba contains predicted probabilities
    y_pred_proba = y_pred_proba.cpu().numpy()
    y_test = y_test.cpu().numpy()
    y_pred = (y_pred_proba >= 0.5).astype(int)  # Convert probabilities to binary predictions

    # Calculate performance metrics
    accuracy = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred)
    auc_score = metrics.roc_auc_score(y_test, y_pred_proba)
    tn, fp, fn, tp = metrics.confusion_matrix(y_test, y_pred).ravel()
    conf_matrix = f'[{tp}, {fp}, {fn}, {tn}]'

    # Store metrics in a dictionary
    eval_score_dict = {
        'accuracy': round(accuracy, 3),
        'precision': round(precision, 3),
        'recall': round(recall, 3),
        'F1_score': round(f1, 3),
        'auc_score': round(auc_score, 3),
        'conf_matrix': conf_matrix
    }
    # Return the y_pred_proba and eval_score_dict
    return (y_pred_proba, eval_score_dict)


def save_model(model=None, root_path='./', itr_tag=None, fold_index=0):
    """
    Save the trained model to the specified path.
    """
    model_path = os.path.join(root_path, 'dataset/postproc_data/pul_result', itr_tag, 'proc_pul')
    PPIPUtils.createFolder(model_path)
    model_name = f'pul_model_fold_{fold_index}.pth'
    torch.save(model.state_dict(), os.path.join(model_path, model_name))
    print(f'model is saved at {os.path.join(model_path, model_name)}')


def perform_pu_learning(root_path='./', itr_tag=None, fold_index=0, train_train_df=None, train_val_df=None, test_df=None, hparams={}):
    """
    Main method to execute PU learning.
    """
    print(f'\n ########## fold_index:{fold_index} :: inside perform_pu_learning() method - Start')
    # prepare the data
    print(f'\n ########## fold_index:{fold_index} :: Preparing the data for the PU learning...\n')
    X_train, y_train, X_val, y_val, X_test, y_test, scaler = pul_prepare_data(train_train_df=train_train_df, train_val_df=train_val_df, test_df=test_df)
    
    # Train the model on the initial dataset
    print(f'\n ########## fold_index:{fold_index} :: Training the model on the initial dataset...\n')
    # Calculate class weights for y_train
    class_weights_y_train = calculate_class_weights(y_train)
    input_size = X_train.shape[1]
    model = train_model(X_train, y_train, X_val, y_val, input_size, num_epochs=hparams['num_epochs'], lr=hparams['lr'], class_weights=class_weights_y_train, hparams=hparams)
    # ### model = train_model_minibatch(X_train, y_train, X_val, y_val, input_size, num_epochs=hparams['num_epochs'], lr=hparams['lr'], class_weights=class_weights_y_train, hparams=hparams, batch_size=512)

    # Identify reliable negatives
    print(f'\n ########## fold_index:{fold_index} :: Identifying reliable negatives...\n')
    # label convention: 0 for negative, 1 for positive and -1 for unlabelled
    # train_train_df contains only positive and unlabelled data
    train_train_df = train_train_df.drop(['prot_id', 'chain_1_seq', 'chain_2_seq'], axis=1)
    train_train_unlabelled_data = train_train_df[train_train_df['label'] == -1].reset_index(drop=True)
    reliable_negatives = identify_reliable_negatives(model, train_train_unlabelled_data, scaler, threshold=hparams['threshold'])

    # Create new training data using reliable_negatives
    print(f'\n ########## fold_index:{fold_index} :: Creating new training data using reliable_negatives...\n')
    train_train_positive_data = train_train_df[train_train_df['label'] == 1].reset_index(drop=True)
    new_train_data = pd.concat([train_train_positive_data, reliable_negatives])
    # Treat reliable-negative data-points as negative.
    new_train_data['label'] = new_train_data['label'].replace(-1, 0)
    # Calculate class weights
    class_weights = calculate_class_weights(new_train_data['label'].values)
    
    # Retrain the model on the new training dataset
    print(f'\n ########## fold_index:{fold_index} :: Retraining the model on the new training dataset...\n')
    model = retrain_model(model, new_train_data, X_val, y_val, scaler, num_epochs=hparams['num_epochs'], class_weights=class_weights)
    # ### model = retrain_model_minibatch(model, new_train_data, X_val, y_val, scaler, num_epochs=hparams['num_epochs'], class_weights=class_weights, batch_size=512)

    # Save the model
    print(f'\n ########## fold_index:{fold_index} :: Saving the model...\n')
    save_model(model=model, root_path=root_path, itr_tag=itr_tag, fold_index=fold_index)

    # Evaluate the model on the test set
    print(f'\n ########## fold_index:{fold_index} :: Evaluating the model on the test set...\n')
    y_pred_proba, eval_score_dict = evaluate_model(model, X_test, y_test)
    print(f'\n ########## fold_index:{fold_index} :: eval_score_dict: \n {eval_score_dict}\n')
    # Save the test_df with the prediction result
    y_pred = (y_pred_proba >= 0.5).astype(int)  # Convert probabilities to binary predictions
    y_pred_proba = np.round(y_pred_proba, 3)
    # Insert y_pred and y_pred_proba as two columns after 'label' column in test_df
    test_df.insert(test_df.columns.get_loc('label') + 1, 'y_pred', y_pred)
    test_df.insert(test_df.columns.get_loc('y_pred') + 1, 'y_pred_proba', y_pred_proba)
    test_df_pred_res_loc = os.path.join(root_path, 'dataset/postproc_data/pul_result', itr_tag, 'proc_pul')
    test_df.to_csv(os.path.join(test_df_pred_res_loc, f'fold_{fold_index}_pred_res.csv'), index=False)

    print(f'\n ########## fold_index:{fold_index} :: inside perform_pu_learning() method - End')
    return eval_score_dict



if __name__ == '__main__':
    root_path = os.path.join('/project/root/directory/path/here')
    root_path = os.path.join('/scratch/pralaycs/Shubh_Working_Remote/PPI_Wkspc/PPI_Code/matpip_pd_prj')

    itr_tag = 'mcmc_fullLen_puFalse_batch5_mutPrcntLen10'
    fold_index = 0
    
    # #################### Hyper-parameters -Start
    hparams = {}
    hparams['num_epochs'] = 20  # Number of epochs for training and retraining
    hparams['lr'] = 0.001  # Learning rate for the optimizer
    hparams['threshold'] = 0.5  # Threshold to identify reliable negatives
    hparams['grid_size'] = 5
    hparams['spline_order'] = 3
    hparams['scale_noise'] = 0.1
    hparams['scale_base'] = 1.0
    hparams['scale_spline'] = 1.0
    # #################### Hyper-parameters -End

    folds_dir_loc = os.path.join(root_path, 'dataset/postproc_data/pul_result', f'{itr_tag}/preproc_pul/preproc_4_pul_folds')
    fold_test_key = f'fold_{fold_index}_test'
    fold_train_train_key = f'fold_{fold_index}_train_train'
    fold_train_validation_key = f'fold_{fold_index}_train_validation'
    train_train_df = pd.read_csv(os.path.join(folds_dir_loc, f'{fold_train_train_key}.csv'))
    train_val_df = pd.read_csv(os.path.join(folds_dir_loc, f'{fold_train_validation_key}.csv'))
    test_df = pd.read_csv(os.path.join(folds_dir_loc, f'{fold_test_key}.csv'))

    fold_result_dict = perform_pu_learning(root_path=root_path, itr_tag=itr_tag, fold_index=fold_index
                                            , train_train_df=train_train_df, train_val_df=train_val_df, test_df=test_df
                                            , hparams=hparams)
    print(f'\n fold_result_dict: \n {fold_result_dict}')
