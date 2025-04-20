# ML
1. NumPy – Numerical computing
python
Copy
Edit
import numpy as np

# Creating a simple array
arr = np.array([1, 2, 3])
print(arr * 2)

2. Pandas – Data manipulation
python
Copy
Edit
import pandas as pd

# Creating a DataFrame
data = {'Name': ['Alice', 'Bob'], 'Score': [90, 85]}
df = pd.DataFrame(data)
print(df)

3. Matplotlib / Seaborn – Visualization
python
Copy
Edit
import matplotlib.pyplot as plt
import seaborn as sns

# Simple plot
sns.set()
scores = [90, 85, 88, 92]
plt.plot(scores)
plt.title("Scores Over Time")
plt.show()



4. Scikit-learn – ML algorithms
python
Copy
Edit
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np

# Sample data
X = np.array([[1], [2], [3], [4]])
y = np.array([2, 4, 6, 8])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

print(model.predict(X_test))


5. TensorFlow / Keras – Deep learning
python
Copy
Edit
import tensorflow as tf
from tensorflow.keras import layers, models

# Simple neural network
model = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(10,)),
    layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')
6. PyTorch – Deep learning (alternative to TensorFlow)
python
Copy
Edit
import torch
import torch.nn as nn

# Simple linear model
model = nn.Linear(10, 1)
input_data = torch.randn(1, 10)
output = model(input_data)
print(output)








