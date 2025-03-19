import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import glob as glob



# Location of data:
location = r'D:\tacc scratch\25_03\18\2bc42c69-0ff8-433b-b5b5-68350caf503c-007'
# location = r'D:\tacc scratch\25_03\18\2c081ff4-795e-407f-997c-4cad6ada1b5b-007'
# location = r'D:\tacc scratch\25_03\18\4c4b2820-f489-40b8-b0e7-3e6294adf21e-007'
# location = r'D:\tacc scratch\25_03\18\6d18ffc0-0bfb-4b73-bb12-de8a601d0572-007'

# Load dakotaTab.out file, which is inside the location + results
dakotaTabPath = os.path.join(location, 'results', 'dakotaTab.out')
dakotaTab = pd.read_csv(dakotaTabPath, sep='\s+')

# Load cal_file.csv which is inside the location
calFilePath = os.path.join(location, 'cal_file.csv')
calFile = pd.read_csv(calFilePath, header=None)

maxForce = max(calFile.iloc[0])
# Get maxForce as the maximum value in the calFile
# maxForce = np.array(calFile).max()

# Get length of the calFile
calFileLength = calFile.shape[1]
print(f'calFileLength: {calFileLength}')

# extract the last calFileLength columns from dakotaTab
dakotaTabForce = dakotaTab.iloc[:, -calFileLength:]

# Get a matrix of residuals by subtracting the calFile to each row of dakotaTabForce
residuals = (dakotaTabForce.values - calFile.values[:, None]) / maxForce
residuals = np.abs(residuals)[0]

# Get the mean of each row
residuals = np.mean(residuals, axis=1)

print(residuals)
print(len(residuals))

# If residual is larger than 1, set the value to 1
residuals[residuals > 1] = 1

# Find the index of the minimum residual
minIndex = np.argmin(residuals)


# Merge data from dakotaTab and all resultsStage files in the results folder

# Get all resultsStage files in the results folder
results_files = glob.glob(os.path.join(location, 'results', 'resultsStage*.csv'))

# Loop through each resultsStage file and merge it with dakotaTab, skipping resultsStage0.csv
for file in results_files:
    if 'resultsStage0.csv' in file:
        continue
    dakotaTabPrior = pd.read_csv(file, sep=',', header=None)
    dakotaTabPrior.columns = dakotaTab.columns[-dakotaTabPrior.shape[1]:]
    dakotaTab = pd.concat([dakotaTab, dakotaTabPrior], ignore_index=True)

dakotaTabForce = dakotaTab.iloc[:, -calFileLength:]

# Compute new residuals with respect to the dakotaTab in the location of minimum residual
minResidualRow = dakotaTab.iloc[minIndex, :]
residuals = (dakotaTabForce.values - minResidualRow[-calFileLength:].values) / maxForce
residuals = np.abs(residuals)

# Get the mean of each row
residuals = np.mean(residuals, axis=1)

dakotaTab['residuals'] = residuals

# Select the parameter columns (excluding the residuals column)
param_columns = ['gamma', 'kappa', 'eta1', 'sig', 'lam', 'mup', 'sigp', 'rsmax', 'alpha', 'alpha1', 'alpha2', 'betam1', 'n', 'kappa_k']


# Transform the parameter columns so that each column is normalized to have mean 0 and std 1
# After this, the values are normalized
# dakotaTab[param_columns] = (dakotaTab[param_columns] - dakotaTab[param_columns].mean()) / dakotaTab[param_columns].std()

# Perform PCA on the parameter columns
pca = PCA(n_components=2)
principal_components = pca.fit_transform(dakotaTab[param_columns])

# Create a DataFrame with the principal components
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

# Add the residuals to the PCA DataFrame
pca_df['residuals'] = dakotaTab['residuals']

# Create a plot of the principal components vs residuals
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pca_df['PC1'], pca_df['PC2'], pca_df['residuals'])
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Residuals')
# plt.show()

'''# Create a new plot with just single parameter vs residual.
for param in param_columns:
    plt.figure()
    plt.scatter(dakotaTab[param], dakotaTab['residuals'])
    plt.xlabel(param)
    plt.ylabel('Residuals')
    plt.title(f'{param} vs Residuals')
    # plt.show()'''

# Create another plot just one pc vs residuals
plt.figure()
plt.scatter(pca_df['PC1'], pca_df['residuals'])
plt.xlabel('Principal Component 1')
plt.ylabel('Residuals')
plt.title('PC1 vs Residuals')
# plt.show()

# Show coefficients of the PCA
print(pca.components_)
print(pca.explained_variance_ratio_)

# Plot the plca.components_ as a bar graph
plt.figure()
plt.bar(range(len(pca.components_[0])), pca.components_[0])
plt.xlabel('Parameters')
plt.ylabel('Coefficient')
plt.title('PCA Coefficients for PC1')
# Add parameter names as x-ticks
plt.xticks(range(len(pca.components_[0])), param_columns, rotation=90)
plt.tight_layout()

# plt.show()

# Do the same as above for PC2
plt.figure()
plt.bar(range(len(pca.components_[1])), pca.components_[1])
plt.xlabel('Parameters')
plt.ylabel('Coefficient')
plt.title('PCA Coefficients for PC2')
# Add parameter names as x-ticks
plt.xticks(range(len(pca.components_[1])), param_columns, rotation=90)
plt.tight_layout()
# plt.show()

# Run linear regression from the original normalized parameters to the residuals


X = dakotaTab[param_columns].values
y = dakotaTab['residuals'].values

# Create a linear regression model
model = LinearRegression()
model.fit(X, y)

# Get the coefficients and intercept
coefficients = model.coef_
intercept = model.intercept_

# Plot the coefficients as a bar graph
plt.figure()
plt.bar(range(len(coefficients)), coefficients)
plt.xlabel('Parameters')
plt.ylabel('Coefficient')
plt.title('Linear Regression Coefficients')
# Add parameter names as x-ticks
plt.xticks(range(len(coefficients)), param_columns, rotation=90)
plt.tight_layout()
# Save this plot as a high-resolution png
plt.savefig('Figures/linear_regression_coefficients.png', dpi=500)
# plt.show()

# Print R2 score of the model
print('R2 score:', model.score(X, y))


# Plot residuals and the linear regression line
plt.figure()
plt.scatter(y, model.predict(X))
plt.xlabel('Residuals')
plt.ylabel('Predicted Residuals')
plt.title('Residuals vs Predicted Residuals')
# Add 1-1 line
plt.plot([0, 0.5], [0, 0.5], 'r--')
plt.tight_layout()
# plt.show()

