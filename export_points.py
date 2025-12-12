import pandas as pd
import numpy as np
from read_from_excel import ExcelReader

def export_points_with_weights(points: np.ndarray, filename="points_with_weights.xlsx"):
    """
    points: numpy array of shape (N, 3)
            [:,0] = X values
            [:,1] = Y values
            [:,2] = Weight values
    filename: output Excel file name
    """

    if points.shape[1] != 3:
        raise ValueError("Input array must have shape (N, 3)")

    df = pd.DataFrame(points, columns=["X", "Y", "Weight"])

    df.to_excel(filename, index=False)

    print(f"Excel file exported successfully: {filename}")


if __name__ == "__main__":
    # Load points from the input Excel
    points = ExcelReader.read_points_with_weights("points.xlsx")

    # Export them to a new file
    export_points_with_weights(points)
