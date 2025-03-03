# Load the failure mode selection model
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Import the merge_data.csv file
data = pd.read_csv('merged_data.csv')

# Define the predictors and the classes
predictors = data[['ar', 'lrr', 'srr', 'alr', 'sdr', 'smr']]
classes = data['FailureType']

# Turn classes into binary values following the following rule:
# if FailureType == 'Flexural' then 1 else 0
classes = classes.apply(lambda x: 1 if x == 'Flexure' else 0)


# Load model
ndParams = ['ar', 'lrr', 'srr', 'alr', 'sdr', 'smr']
filename = 'logistic_regression_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))

# Create a 3d surface plot using the logistic regression model
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ar = np.linspace(0, 8.0, 100)
smr = np.linspace(0, 4.0, 100)
ar, smr = np.meshgrid(ar, smr)

# Define the rest of the values as constant values
x = ar
y = smr

# Predict the probability of flexure
lrr = 0.1
srr = 0.1
alr = 0.1
sdr = 0.5

z = loaded_model.predict_proba(pd.DataFrame({ndParams[0]: x.flatten(), ndParams[1]: y.flatten(), ndParams[2]: srr, ndParams[3]: alr, ndParams[4]: sdr, ndParams[5]: lrr}))[:, 1].reshape(x.shape)
ax.plot_surface(x, y, z, cmap='viridis')
ax.set_xlabel('ar')
ax.set_ylabel('smr')
ax.set_zlabel('Prob. of Flexure Failure')

# Add scatter with predictors and classes, color by class value
scatter = ax.scatter(predictors['ar'], predictors['smr'], classes, c=classes, cmap='bwr', marker='o')
legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
ax.add_artist(legend1)

# Add limits in x and y
ax.set_xlim([0, 8.0])
ax.set_ylim([0, 4.0])

# Set angle to -45 degrees
ax.view_init(elev=45, azim=135)
plt.show()