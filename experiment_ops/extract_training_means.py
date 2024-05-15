import numpy as np
import pandas as pd

train = pd.read_csv("../bot/data/no_missing_training_data.csv", parse_dates=["timestamp"])
train_means = train.mean(numeric_only=False)
# series to one-row dataframe
train_means = pd.DataFrame(train_means).T

train_means.to_csv("../bot/data/training_means.csv", index=False)