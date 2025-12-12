import pandas as pd
import numpy as np

df = pd.read_excel("points.xlsx")

points = df[['x', 'y']].to_numpy()

print(points)
