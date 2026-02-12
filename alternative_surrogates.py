# Alternative models for surrogate (not GP)
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

import os

from gpPredict import *
from get_bw_params import *


def train_model(model_type, X_train, y_train):
    if model_type == 'RandomForest':
        model = RandomForestRegressor(n_estimators=500, random_state=42)
    elif model_type == 'DecisionTree':
        model = DecisionTreeRegressor(random_state=42)
    elif model_type == 'LinearRegression':
        model = LinearRegression()
    elif model_type == 'NeuralNetwork':
        from sklearn.neural_network import MLPRegressor
        model = MLPRegressor(hidden_layer_sizes=(100,20,20,20,20), max_iter=1000, random_state=42)
    else:
        raise ValueError("Unsupported model type")
    
    # Train the model
    model.fit(X_train, y_train)

    return model


if __name__ == "__main__":

    # Define parameter names (predictors and outputs)
    predictors = ['ar', 'lrr', 'srr', 'alr', 'sdr', 'smr']
    outputs = ['gamma','kappa','eta1','sig','lam','mup','sigp','rsmax','alpha','alpha1','alpha2','betam1','n','kappa_k']

    out_path = 'add_model_results'
    
    # Load training data from 
    main_data_path = f'gp_training_data\\processed\\gpModelFlexure'

    # Load the data_split.csv
    print(f"Loading data from {main_data_path}...")
    data_split = pd.read_csv(os.path.join(main_data_path, 'data_split.csv'))
    print(f"Loaded {len(data_split)} samples")

    # Available model types
    model_types = ['RandomForest', 'DecisionTree', 'LinearRegression', 'NeuralNetwork']
    splits = sorted(data_split['split'].unique())
    
    # Step #1: Try loading the predictions from existing CSVs
    prediction_dfs = []

    for model_type in model_types:
        prediction_file = os.path.join(out_path, f'predictions_{model_type}.csv')

        if not os.path.exists(prediction_file):
            print("Prediction files not found. Training models..")
            print(f"\n{'='*50}")
            print(f"Training {model_type} model")
            print(f"{'='*50}")
            
            # Create a dataframe to store predictions for all splits
            predictions_df = pd.DataFrame(columns=['UniqueId'] + predictors + outputs + ['split'])

            for test_split in splits:
                # Train on all splits except test_split
                X_train = data_split[predictors][data_split['split'] != test_split]
                y_train = data_split[outputs][data_split['split'] != test_split]
                
                model = train_model(model_type, X_train, y_train)
                
                # Test on current split
                X_test = data_split[predictors][data_split['split'] == test_split]
                y_test = data_split[outputs][data_split['split'] == test_split]
                y_pred = model.predict(X_test)

                print(f"Split {test_split}: trained on {len(X_train)} samples, tested on {len(X_test)} samples")

                # Store predictions adding them to the predictions_df
                split_predictions = pd.DataFrame(y_pred, columns=outputs)
                split_predictions['UniqueId'] = data_split['UniqueId'][data_split['split'] == test_split].values
                split_predictions[predictors] = X_test.reset_index(drop=True)
                split_predictions['split'] = test_split
                predictions_df = pd.concat([predictions_df, split_predictions], ignore_index=True)

            # Save predictions to CSV
            # If the output directory does not exist, create it
            if not os.path.exists(out_path):
                os.makedirs(out_path)

            output_file = os.path.join(out_path, f'predictions_{model_type}.csv')
            predictions_df.to_csv(output_file, index=False)

        else:
            print(f"Loading existing predictions for {model_type} from {prediction_file}")
            predictions_df = pd.read_csv(prediction_file)

        prediction_dfs.append((model_type, predictions_df))

    print("All predictions loaded/trained.")

    # Plot the predictions and the true values for comparison and 1-1 line
    #for model_type, predictions_df in prediction_dfs:
    #    print(f"\nPlotting results for {model_type}...")
    #    for output in outputs:
    #        plt.figure(figsize=(8, 8))
    #        plt.scatter(data_split[output], predictions_df[output], alpha=0.5)
    #        plt.plot([data_split[output].min(), data_split[output].max()], 
    #                 [data_split[output].min(), data_split[output].max()], 'r--')
    #        plt.xlabel('True Values')
    #        plt.ylabel('Predicted Values')
    #        plt.title(f'{model_type} Predictions vs True Values for {output}')
    #        plt.grid(True)
    #        plt.axis('equal')
    #        plt.show()
    

    # ---
    # Part 2: get the predictions from the GP model for each split
    # ---

    # Try loading the predictions from existing CSV
    if not os.path.exists(os.path.join(out_path, f'predictions_GP.csv')):
        print("GP Prediction file not found. Getting GP predictions...")

        surr_path = 'gp_training_data/processed'

        predictions_df_gp = pd.DataFrame(columns=['UniqueId'] + predictors + outputs + ['split'])

        for test_split in splits:
            # Load the data_split.csv
            print(f"\nGetting GP predictions for split {test_split}...")

            # Test on current split
            X_test = data_split[predictors][data_split['split'] == test_split]
            
            # For each row on X_test, get the GP predictions
            for idx, row in X_test.iterrows():
                ndParams = row.values.tolist()
                print(f"\nParameters: {ndParams}")
                split_code = f'split_{str(test_split).zfill(2)}'
                predParams, predErr, failureMode = get_BW_params(ndParams, surr_path, split=split_code, mode='simple')
                print(f"Predicted BW Params: {predParams}")

                # Store the values in the predictions_df_gp
                pred_dict = {'UniqueId': data_split['UniqueId'][idx], 'split': test_split}
                for i, param in enumerate(predictors):
                    pred_dict[param] = ndParams[i]

                for i, param in enumerate(outputs):
                    pred_dict[param] = predParams[i]
                predictions_df_gp = predictions_df_gp.append(pred_dict, ignore_index=True)

        print("GP predictions obtained.")

        # Save GP predictions to CSV
        gp_output_file = os.path.join(out_path, f'predictions_GP.csv')
        predictions_df_gp.to_csv(gp_output_file, index=False)
        print("GP predictions saved.")
    
    else:
        print("Loading existing GP predictions...")
        predictions_df_gp = pd.read_csv(os.path.join(out_path, f'predictions_GP.csv'))
        print("GP predictions loaded.")

    # ---
    # Part 3: Compare selected models using NRMSE heatmap
    # ---
    print("\nComparing selected models with NRMSE...")

    from sklearn.metrics import mean_squared_error
    import seaborn as sns

    # Select only GP, RandomForest, and NeuralNetwork for comparison
    selected_models = [('RandomForest', prediction_dfs[0][1]), 
                       ('NeuralNetwork', prediction_dfs[3][1]), 
                       ('GP', predictions_df_gp)]

    # Calculate NRMSE for each model and output
    nrmse_results = []
    for model_type, predictions_df in selected_models:
        nrmse_values = []
        for output in outputs:
            y_true = data_split[output].values
            y_pred = predictions_df[output].values
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            y_range = y_true.max() - y_true.min()
            nrmse = rmse / y_range if y_range != 0 else 0
            nrmse_values.append(nrmse)
        nrmse_results.append(nrmse_values)
    
    # Create LaTeX-style labels for output parameters
    output_labels = {
        'gamma': r'$\gamma$',
        'kappa': r'$\kappa$',
        'eta1': r'$\eta_1$',
        'sig': r'$\sigma$',
        'lam': r'$\lambda$',
        'mup': r'$\mu_p$',
        'sigp': r'$\sigma_p$',
        'rsmax': r'$r_{s,max}$',
        'alpha': r'$\alpha$',
        'alpha1': r'$\alpha_1$',
        'alpha2': r'$\alpha_2$',
        'betam1': r'$\beta_{m1}$',
        'n': r'$n$',
        'kappa_k': r'$\kappa_k$'
    }
    
    latex_labels = [output_labels[out] for out in outputs]
    
    # Create heatmap
    nrmse_matrix = np.array(nrmse_results)
    model_names = ['RF', 'NN', 'GP']  # Use acronyms for cleaner labels
    
    plt.figure(figsize=(12, 4))
    sns.heatmap(nrmse_matrix, annot=True, fmt='.4f', cmap='RdYlGn_r', 
                xticklabels=latex_labels, yticklabels=model_names, cbar_kws={'label': 'NRMSE'})
    plt.title('NRMSE (Normalized RMSE) Comparison: GP vs RandomForest vs NeuralNetwork', fontsize=12)
    plt.xlabel('Output Parameters', fontsize=11)
    plt.ylabel('Model', fontsize=11)
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("\nNRMSE Summary (lower is better):")
    print("-" * 70)
    full_names = ['RandomForest', 'NeuralNetwork', 'GP']
    for i, (short_name, full_name) in enumerate(zip(model_names, full_names)):
        mean_nrmse = np.mean(nrmse_matrix[i])
        print(f"{full_name:20s} ({short_name}) - Mean NRMSE: {mean_nrmse:.4f}, Min: {np.min(nrmse_matrix[i]):.4f}, Max: {np.max(nrmse_matrix[i]):.4f}")