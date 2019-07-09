import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class Analysis:

    @property
    def model(self):
        return self._model

    @property
    def fitted(self):
        return self._fitted
    
    @property
    def comps(self):
        return self._comps
    
    @property
    def maps(self):
        return self._maps
    
    @property
    def gridsize(self):
        return self._gridsize

    @property
    def samp_flags(self):
        return self._samp_flags

    @property
    def maxes(self):
        return self._maxes

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

    def plot(self, spacer=None, norm=None, wfits=False, colorbar=False, justmaps=0):

        numSamples = self._maps.columns.levels[0].shape[0]
        numComps = self._comps.shape[0]
        numMeas = self._comps.columns.levels[0].shape[0]
        numVars = self._comps.columns.levels[1].shape[0]
        fitted = self._fitted

        plotrows = numComps

        if justmaps:
            plotcols = numSamples
        else:
            plotcols = numSamples + numVars*numMeas

        i = 1

        for samp in self._maps.columns.levels[0]:

            j = 0

            #flags = self._samp_flags[samp]
            grid = self._gridsize[samp]

            for comp in self._maps[samp]:

                #newmap = np.empty(grid*grid)
                #newmap[flags] = np.inf
                #newmap[~flags] = self._maps[samp][comp].values
                newmap = self._maps[samp][comp].values

                mapdata = np.reshape(newmap, [grid, grid])

                sub = plt.subplot(plotrows, plotcols, i + j)
                sub.axis('off')

                if norm is 'ZeroToOne':
                    plot = sub.imshow(mapdata, cmap='jet', vmin=0, vmax=1)
                elif norm is 'OneToOne':
                    plot = sub.imshow(mapdata, cmap='jet', vmin=-1, vmax=1)
                else:
                    plot = sub.imshow(mapdata, cmap='jet')
                    plt.colorbar(plot, ax=sub)

                if j == 0:
                    sub.set_title(samp)

                j = j + plotcols

            i = i + 1

        if norm is not None and colorbar:
            plt.colorbar(plot, ax=sub)

        i = 1

        if justmaps == 0:
            for meas in self._comps.columns.levels[0]:

                for var in self._comps.columns.levels[1]:

                    j = 0

                    for comp in np.arange(0, numComps):

                        compdata = self._comps[meas][var].xs(comp)[:]
                        plotdata = compdata.values

                        cols = self._comps.columns.values
                        xax = np.asarray([list(t) for t in zip(*cols)])[-1, :]

                        xax = xax.astype(float)
                        xax = xax[0:len(compdata)]

                        if spacer is not None:
                            newxax = np.full(spacer.shape, np.inf)
                            newdata = np.full(spacer.shape, np.inf)
                            newxax[spacer] = xax
                            newdata[spacer] = plotdata

                            xax = newxax
                            plotdata = newdata

                            if wfits is True and var == 'Amp':
                                newdata = np.full(spacer.shape, np.inf)
                                newdata[spacer] = fitted.loc[comp, :]
                                fit_data = newdata

                        if wfits is True and var == 'Amp':
                            sub = plt.subplot(plotrows, plotcols, i + j + numSamples)
                            sub.plot(xax, plotdata, '.k')
                            sub.plot(xax, fit_data, '-r')
                        else:
                            sub = plt.subplot(plotrows, plotcols, i + j + numSamples)
                            sub.plot(xax, plotdata)


                        if j == 0:
                            sub.set_title(str(meas)+': ' + str(var))

                        j = j + plotcols

                    i = i + 1