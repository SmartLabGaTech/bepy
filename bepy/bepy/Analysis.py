import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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

    @property
    def samp_flags(self):
        return

    @property
    def maxes(self):
        return

    def __init__(self, model=None, fitted=None, comps=None, maps=None, gridSize=50, samp_flags=None):
        self._gridsize = gridSize
        self._model = model
        self._fitted = fitted

        self._maps = maps
        self._comps = comps
        self._samp_flags = samp_flags
        self._maxes = {}

    def normalize(self, norm_type='ZeroToOne'):

        idx = pd.IndexSlice

        if norm_type is 'ZeroToOne':
            for i in self._maps.columns.get_level_values(1).unique():
                self._maxes[i] = self._maps.loc[:, idx[:, i]].max().max()

                self._maps.loc[:, idx[:, i]] = self._maps.loc[:, idx[:, i]]/self._maxes[i]
                self._comps.loc[i, :] = self._comps.loc[i, :]*self._maxes[i]
        else:
            pass

    def plot(self, spacer=None, norm=None):

        numSamples = self._maps.columns.levels[0].shape[0]
        numComps = self._comps.shape[0]
        numMeas = self._comps.columns.levels[0].shape[0]
        numVars = self._comps.columns.levels[1].shape[0]

        plotrows = numComps
        plotcols = numSamples + numVars*numMeas

        i = 1

        for samp in self._maps.columns.levels[0]:

            j = 0

            flags = self._samp_flags[samp]
            grid = self._gridsize[samp]

            for comp in self._maps[samp]:

                newmap = np.empty(flags.shape)
                newmap[flags] = np.inf
                newmap[~flags] = self._maps[samp][comp].values

                mapdata = np.reshape(newmap, [grid, grid])

                sub = plt.subplot(plotrows, plotcols, i + j)

                if norm is 'ZeroToOne':
                    plot = sub.imshow(mapdata, cmap='jet', vmin=0, vmax=1)
                else:
                    plot = sub.imshow(mapdata, cmap='jet')

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