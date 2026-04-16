import pandas as pd

def load_data(path):
    data = pd.read_csv(path)
    X = data.drop("Flood_Label", axis=1)
    y = data["Flood_Label"]
    return X, y
