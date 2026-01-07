import pandas as pd
import numpy as np

class ExcelReader:
    @staticmethod
    def read_points_with_weights(filename="points.xlsx") -> np.ndarray:
        """
        Reads points with weights from an Excel file (columns: x, y, weight).

        Returns:
            numpy array of shape (N, 3)
            [:,0] = X values
            [:,1] = Y values
            [:,2] = Weight values
        """
        df = pd.read_excel(filename)

        # Normalize column names
        df.columns = [col.lower().strip() for col in df.columns]

        required = ["x", "y", "weight"]
        if not all(col in df.columns for col in required):
            raise ValueError(f"Excel file must contain columns: {required}. Found: {df.columns.tolist()}")

        points = df[["x", "y", "weight"]].to_numpy()

        return points

    def read_points(filename="points.xlsx") -> np.ndarray:
        """
        Reads points from an Excel file (columns: x, y).

        Returns:
            numpy array of shape (N, 2)
            [:,0] = X values
            [:,1] = Y values
        """
        df = pd.read_excel(filename)

        # Normalize column names
        df.columns = [col.lower().strip() for col in df.columns]

        required = ["x", "y"]
        if not all(col in df.columns for col in required):
            raise ValueError(
                f"Excel file must contain columns: {required}. "
                f"Found: {df.columns.tolist()}"
            )

        points = df[["x", "y"]].to_numpy(dtype=float)

        return points

if __name__ == "__main__":
    points = ExcelReader.read_points_with_weights("points.xlsx")

    print("Loaded points:")
    print(points)
    print("Shape:", points.shape)
