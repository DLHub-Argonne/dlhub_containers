from sklearn.datasets import load_iris
from sklearn.svm import SVC
import pickle as pkl
import pandas as pd

# Load the data
X, y = load_iris(return_X_y=True)

# Make the model
model = SVC(kernel='linear', C=1, probability=True)
model.fit(X, y)
print('Trained a SVC model')

# Save the model using pickle
with open('model.pkl', 'wb') as fp:
    pkl.dump(model, fp)
print('Saved model to disk')
