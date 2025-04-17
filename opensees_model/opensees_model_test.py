# This is an opensees model

import openseespy.opensees as ops

# Define the model
ops.wipe()  # Clear any existing model

# Create a new model
ops.model('basic', '-ndm', 2, '-ndf', 3)  # 2D model with 3 degrees of freedom per node (x, y, rotation)