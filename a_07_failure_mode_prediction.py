import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Set matplotlib text style to latex
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

figures_folder = 'Figures'

# Load the dataset
column_data = pd.read_csv('gp_training_data/raw/DataAll_NDonly.csv')

# Turn the Flexure-Shear values into Shear
column_data['FailureType'] = column_data['FailureType'].apply(lambda x: 'Shear' if x == 'Flexure-Shear' else x)

# Run lasso regression using ['ar', 'lrr', 'srr', 'alr', 'sdr', 'smr'] as features and 'FailureType' as the target variable
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split

# Normalize the input features 
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
column_data[['ar', 'lrr', 'srr', 'alr', 'sdr', 'smr']] = scaler.fit_transform(column_data[['ar', 'lrr', 'srr', 'alr', 'sdr', 'smr']])

X = column_data[['ar', 'lrr', 'srr', 'alr', 'sdr', 'smr']]
# Get the inverse of each feature
y = column_data['FailureType']

# Turn y values into integers from categorical
y = y.astype('category').cat.codes

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)

# Print the coefficients
print(lasso.coef_)
print(lasso.intercept_)
print(lasso.score(X_test, y_test))


'''# Plot the coefficients as bars
plt.bar(X.columns, lasso.coef_)
plt.show()'''

# Conclusion:
# The last two features have the most impact, once they have been normalized

# Denormalize the data

column_data[['ar', 'lrr', 'srr', 'alr', 'sdr', 'smr']] = scaler.inverse_transform(column_data[['ar', 'lrr', 'srr', 'alr', 'sdr', 'smr']])

'''from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

X = column_data[['sdr', 'smr']]
y = column_data['FailureType']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

dt = DecisionTreeClassifier(max_depth=3)
dt.fit(X_train, y_train)

y_pred = dt.predict(X_test)
print(accuracy_score(y_test, y_pred))

# Print the tree
from sklearn.tree import export_text
r = export_text(dt, feature_names=['sdr', 'smr'])
print(r)

# Pretty-print the tree
from sklearn.tree import plot_tree
plt.figure(figsize=(20, 10))
plot_tree(dt, feature_names=['sdr', 'smr'], filled=True)
plt.show()'''

## Train a logistic regression using the first and the last features
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

parameters = ['ar', 'smr']
X = column_data[parameters]
y = column_data['FailureType']

# Change the target variable to integers
y = y.astype('category').cat.codes

# Train 10 times and do cross-validation
lr = LogisticRegression()
scores = cross_val_score(lr, X, y, cv=10)

print('Cross-validation scores:', scores)
print('Mean cross-validation score:', scores.mean())

# Fit the model on the entire dataset
lr.fit(X, y)

print('Model Information:')
print('Model coefficients', lr.coef_)
print('Model intercept', lr.intercept_)

# Plot the probability of FailureType being 1 in a 3d plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Create color code depending on the actual target variable
colors = {0: 'b', 1: 'r'}
column_data['color'] = y.apply(lambda x: colors[x])

# Scatter plot showing actual FailureType
ax.scatter(column_data[parameters[0]], column_data[parameters[1]], y, c=column_data['color'])

# Create a meshgrid for the surface plot
param1_range = np.linspace(column_data[parameters[0]].min(), column_data[parameters[0]].max(), 50)
param2_range = np.linspace(column_data[parameters[1]].min(), column_data[parameters[1]].max(), 50)
param1_grid, param2_grid = np.meshgrid(param1_range, param2_range)
X_grid = np.c_[param1_grid.ravel(), param2_grid.ravel()]
proba_grid = lr.predict_proba(X_grid)[:, 1].reshape(param1_grid.shape)

# Surface plot
ax.plot_surface(param1_grid, param2_grid, proba_grid, cmap='viridis', edgecolor='none', alpha=0.5)

# Add labels
ax.set_xlabel(parameters[0])
ax.set_ylabel(parameters[1])
ax.set_zlabel('P(Shear Failure)')
plt.show()

# Plot the decision boundary
plt.xlabel('Aspect Ratio ($A_R$)')
plt.ylabel('Strength Ratio $V_p/V_s$')

# Create a meshgrid for the decision boundary
param1_range = np.linspace(column_data[parameters[0]].min(), column_data[parameters[0]].max(), 500)
param2_range = np.linspace(column_data[parameters[1]].min(), column_data[parameters[1]].max(), 500)
param1_grid, param2_grid = np.meshgrid(param1_range, param2_range)
X_grid = np.c_[param1_grid.ravel(), param2_grid.ravel()]
y_grid = lr.predict(X_grid).reshape(param1_grid.shape)

# Plot the decision boundary
plt.contourf(param1_grid, param2_grid, y_grid, alpha=0.2, cmap='jet', levels=2)

# Plot points with rectangle (square) and circle markers per failure type
marker_map = {'Flexure': 's', 'Shear': 'o'}
color_map = {'Flexure': 'b', 'Shear': 'r'}

for t in column_data['FailureType'].unique():
    df_t = column_data[column_data['FailureType'] == t]
    plt.scatter(df_t[parameters[0]],
                df_t[parameters[1]],
                c=color_map.get(t, 'k'),
                edgecolor=color_map.get(t, 'k'),
                marker=marker_map.get(t, 'o'),
                alpha=0.6,
                label=t)

# Legend
plt.legend()

# Set axes square
plt.xlim([1, 8])
plt.ylim([0, 4])
plt.savefig(figures_folder + r'/failure_mode_decision_boundary.pdf', format='pdf')

plt.show()

# Save tihs figure

# Save model as a pickle file
import pickle

with open('log_reg_model.pkl', 'wb') as f:
    pickle.dump(lr, f)


# Do pairplot of the data
import seaborn as sns
# Create pairplot
pairplot = sns.pairplot(column_data[['ar', 'lrr', 'srr', 'alr', 'sdr', 'smr', 'Type']], 
                        hue='Type', 
                        plot_kws={'alpha': 0.5},
                        markers=['s', 'o'],
                        palette={'Rectangular': 'blue', 'Spiral': 'black'},
                        diag_kind='hist', 
                        height=1.0)

# Add space between the plots (like tight_layout)
plt.subplots_adjust(wspace=0.2, hspace=0.2)

# Set ranges of values per parameter
for ax in pairplot.axes.flatten():
    if ax.get_xlabel() == 'ar':
        ax.set_xlim(0, 8)
        ax.set_xlabel('$AR$')
    if ax.get_ylabel() == 'ar':
        ax.set_ylim(0, 8)
        ax.set_ylabel('$AR$')
    if ax.get_xlabel() == 'lrr':
        ax.set_xlim(0, 1.2)
        ax.set_xlabel('$LRR$')
    if ax.get_ylabel() == 'lrr':
        ax.set_ylim(0, 1.2)
        ax.set_ylabel('$LRR$')
    if ax.get_xlabel() == 'srr':
        ax.set_xlim(0, 0.6)
        ax.set_xlabel('$TRR$')
    if ax.get_ylabel() == 'srr':
        ax.set_ylim(0, 0.6)
        ax.set_ylabel('$TRR$')
    if ax.get_xlabel() == 'alr':
        ax.set_xlim(0, 1.2)
        ax.set_xlabel('$ALR$')
    if ax.get_ylabel() == 'alr':
        ax.set_ylim(0, 1.2)
        ax.set_ylabel('$ALR$')
    if ax.get_xlabel() == 'sdr':
        ax.set_xlim(0, 10)
        ax.set_xlabel('$TSR$')
    if ax.get_ylabel() == 'sdr':
        ax.set_ylim(0, 10)
        ax.set_ylabel('$TSR$')
    if ax.get_xlabel() == 'smr':
        ax.set_xlim(0, 5)
        ax.set_xlabel('$SSR$')
    if ax.get_ylabel() == 'smr':
        ax.set_ylim(0, 5)
        ax.set_ylabel('$SSR$')

plt.show()
# plt.savefig(figures_folder + r'/pairplot.pdf', format='pdf')
