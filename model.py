import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pickle

# Read dataset
df = pd.read_csv("Cab_Data.csv")

def encoder(firm):
  if firm == "Pink Cab":
    return(0)
  else:
    return(1)

df["Company"]=df["Company"].apply(encoder)

x = df[["KM Travelled", "Cost of Trip","Company"]]
y = df["Price Charged"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=50)

# Instantiate 
classifier = RandomForestRegressor(n_estimators = 300, max_features = 'sqrt', max_depth = 7, random_state = 18)

# Fit
classifier.fit(x_train, y_train)

# Create .pkl file
pickle.dump(classifier, open("model.pkl", "wb"))