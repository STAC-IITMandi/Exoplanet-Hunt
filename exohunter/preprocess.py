import numpy as np
import pandas as pd

__all__ = ['reader']


def reader(location):
    """
    Returns prime numbers up to imax.

    Parameters
    ----------
    location: string
        The path to the file containing desired data.

    Returns
    -------
    result: tuple
        (Labels, Row Data) for training.
    """
    df = pd.read_csv(location)
    labels = df.iloc[:, 0] - 1
    fluxes = df.iloc[:, 1:]
    fluxes = (fluxes - fluxes.mean())/(fluxes.std())
    return labels, fluxes
