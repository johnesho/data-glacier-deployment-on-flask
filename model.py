# import libraries 
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# load the dataset
df = pd.read_csv("Iris.csv")
# print(df)

# select independent and dependent variables
X = df[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]]
y = df["Species"]

# split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

# instantiate model
classifier = RandomForestClassifier()

# fit model
classifier.fit(X_train, y_train)

# create pickle file
pickle.dump(classifier, open("model.pkl", "wb"))