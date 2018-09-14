import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.decomposition as learn
from bepy import LineMeasurement, GridMeasurement, Sample, BEData

class Analysis():

    @property
    def model(self):
        return

    @property
    def fitted(self):
        return
    
    @property
    def comps(self):
        return
    
    @property
    def maps(self):
        return
    
    @property
    def gridsize(self):
        return

    def __init__(self, model=None, fitted=None, comps=None, maps=None, gridSize=50):
        self._gridsize = gridSize
        self._model = model
        self._fitted = fitted

        self._maps = maps
        self._comps = comps

    def plot(self, spacer=None):

        numSamples = self._maps.columns.levels[0].shape[0]
        numComps = self._comps.shape[0]
        numMeas = self._comps.columns.levels[0].shape[0]
        numVars = self._comps.columns.levels[1].shape[0]

        plotrows = numComps
        plotcols = numSamples + numVars*numMeas

        i = 1

        for samp in self._maps.columns.levels[0]:

            j = 0

            for comp in self._maps[samp]:

                mapdata = np.reshape(self._maps[samp][comp].values,[self._gridsize,self._gridsize])

                sub = plt.subplot(plotrows, plotcols, i + j)
                plot = sub.imshow(mapdata, cmap='nipy_spectral')
                plt.colorbar(plot, ax=sub)

                if j == 0:
                    sub.set_title(samp)

                j = j + plotcols

            i = i + 1

        i = 1

        for meas in self._comps.columns.levels[0]:

            for var in self._comps.columns.levels[1]:

                j = 0

                for comp in np.arange(0, numComps):

                    compdata = self._comps[meas][var].xs(comp)[:]
                    plotdata = compdata.values
                    
                    cols = self._comps.columns.values
                    xax = np.asarray([list(t) for t in zip(*cols)])[5,:]
                    
                    xax = xax.astype(float)
                    
                    if spacer is not None:
                        newxax = np.full(spacer.shape, np.inf)
                        newdata = np.full(spacer.shape, np.inf)
                        newxax[spacer] = xax
                        newdata[spacer] = plotdata
                        
                        xax = newxax
                        plotdata = newdata
                    
                    sub = plt.subplot(plotrows, plotcols, i + j + numSamples)
                    sub.plot(xax, plotdata)

                    if j == 0:
                        sub.set_title(str(meas)+': ' + str(var))

                    j = j + plotcols

                i = i + 1