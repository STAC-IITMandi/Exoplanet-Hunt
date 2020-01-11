import numpy as np
def ReadLebels(location):
        '''This function returns a list of labels.
        In which True means star has confirmed exoplanet and False means no exoplabet.'''
        Labels = []
        with open(location, 'r') as data:
                for line in list(data)[1:]:
                        Labels.append(bool(int(line.split(',')[0]) - 1))
        return np.array(Labels)	
def ReadFluxes(location):
        '''This function returns a list of fluxes.
        In which each element is a list having flux of a star (as a float value) at different intervals of time.'''
        Fluxes = []
        with open(location, 'r') as data:
                for star in list(data)[1:]:
                        Fluxes.append(np.array(star.split(',')[1:], float))
        return np.array(Fluxes)
