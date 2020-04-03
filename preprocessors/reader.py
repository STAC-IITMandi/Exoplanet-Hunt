# importing library(ies)
import numpy as np

def ReadData(location):
    '''This function returns a list of labels and fluxes.
    In labels '1' means star has confirmed exoplanet and '0' means no exoplabet.'''

    # reading data from .csv file
    df = np.genfromtxt(location, delimiter=",", skip_header=1)
    #our data has labels in column '1', and corresponding fluxes are in respective rows
    labels = df[:,0]
    #getting flux data
    fluxes = np.delete(df, 0, axis=1)
    labels-=1
    fluxes = np.interp(fluxes, (fluxes.min(), fluxes.max()), (0, 1))
    return labels, fluxes
