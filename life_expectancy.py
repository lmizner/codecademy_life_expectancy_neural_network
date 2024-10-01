import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense
from tensorflow.keras.optimizers import Adam

# Load dataset
dataset = pd.read_csv('life_expectancy.csv')
print(dataset.head())
print(dataset.describe())

# Drop Country column
dataset = dataset.drop(['Country'], axis = 1)

# Split into lables and features 
# "Life Expectancy" used for labels
labels = dataset.iloc[:, -1]
features = dataset.iloc[:, 0:-1]

# One-Hot-Encode categorical columns
features = pd.get_dummies(features)

# Split into train and test sets
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.20, random_state = 42)

# Standardize numerical features
numerical_features = features.select_dtypes(include = ['float', 'int64']) 
numerical_columns = numerical_features.columns

ct = ColumnTransformer([("only numeric", StandardScaler(), numerical_columns)], remainder = 'passthrough')

# Fit/Transform training data
features_trained_scaled = ct.fit_transform(features_train)

# Transform testing data
features_test_scaled = ct.transform(features_test)

# Build Neural Network model
model = Sequential()
input_layer = InputLayer(input_shape = (dataset.shape[1], ))
model.add(input_layer)
model.add(Dense(64, activation = 'relu'))
model.add(Dense(1))

print(model.summary())

# Initialize optimizer and compile the model
optimizer = Adam(learning_rate = 0.01)
model.compile(loss = 'mse', metrics = ['mae'], optimizer = optimizer)

# Fit and evaluate model
model.fit(features_trained_scaled, labels_train, epochs = 40, batch_size = 1, verbose = 1)
mse_result, mae_result = model.evaluate(features_test_scaled, labels_test, verbose = 0)
print(mse_result, mae_result)




