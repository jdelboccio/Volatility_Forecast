import plotly.express as px
import numpy as np

def plot_3d_factors():
    """
    Generates a 3D scatter plot of factor weighting.
    """
    factors = ["Fundamentals", "Valuation", "Sentiment"]
    weights = np.random.rand(len(factors))  # Ensure same length

    fig = px.scatter_3d(
        x=weights, y=weights, z=weights,
        text=factors,
        labels={"x": "Fundamentals", "y": "Valuation", "z": "Sentiment"},
        title="Factor Weighting in Model",
        size=[10] * len(factors)
    )
    return fig
