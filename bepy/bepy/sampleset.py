import pandas as pd
import numpy as np
from . import analysis


class SampleSet:

    @property
    def samples(self):
        return self._samples

    @property
    def analysis(self):
        return self._analysis

    @property
    def gridsize(self):
        return self._gridsize

    def __init__(self):
        self._samples = {}
        self._analysis = 0
        self._samp_acq_flags = {}
        self._gridsize = {}

    def addsample(self, samp, sampName, gridsize):
        self._samp_acq_flags[sampName] = samp._meas_acq_flags
        self._samples[sampName] = samp
        self._gridsize[sampName] = gridsize

    def GetSampStack(self, sampstack=None, measstack=None, varstack=['Amp', 'Phase', 'Res', 'Q'], inout=0.0,
                     plotGroup=None, insert=None, clean=False, ignore=False):

        if sampstack is None:
            sampstack = self._samples.keys()

        newInd = 0
        oldInd = 0
        stackreturn = 0

        samp_flags = {}

        for samp in sampstack:

            sample = self._samples[samp]

            if measstack is None:
                measstack = sample._gridmeasurements.keys()

            if clean:
                data, flags = sample.GetMeasStack(measstack=measstack, varstack=varstack, inout=inout,
                                                  plotGroup=plotGroup, insert=insert, clean=clean)
                samp_flags[samp] = flags
            else:
                data = sample.GetMeasStack(measstack=measstack, varstack=varstack, inout=inout, plotGroup=plotGroup,
                                           insert=insert, clean=clean)
                samp_flags[samp] = np.full(sample._gridmeasurements[measstack[0]]._data.shape[0], False)

            try:
                newInd = np.hstack([newInd, np.repeat(samp, data.shape[0])])
                oldInd = np.hstack([oldInd, data.index.values])
                stackreturn = pd.concat([stackreturn, data], axis=0, ignore_index=ignore)
            except TypeError:
                newInd = np.repeat(samp, data.shape[0])
                oldInd = data.index.values
                stackreturn = data
            except Exception as e:
                print(e)

        ind = np.vstack([newInd, oldInd])
        tuples = list(zip(*ind))

        index = pd.MultiIndex.from_tuples(tuples, names=['Sample', 'Point'])

        return pd.DataFrame(stackreturn.values, index=index, columns=stackreturn.columns), samp_flags

    def fit(self, model, sampstack=None, measstack=None, varstack=['Amp', 'Phase', 'Res', 'Q'], inout=0.0,
            plotGroup=None, insert=None, clean=False, umap=False, targets=None, custom=None, custom_flags=None, custom_ind=None, custom_cols=None):

        if custom is None:
            data, samp_flags = self.GetSampStack(sampstack, measstack, varstack, inout, plotGroup, insert, clean=clean)
            inds = data.index
            cols = data.columns
            indata = data.values[:]
        else:
            samp_flags = custom_flags
            indata = custom
            inds = custom_ind
            cols = custom_cols

        indata[np.where(indata==np.inf)] = 0
        indata[np.where(indata==np.nan)] = 0

        if umap:
            maps = model.fit_transform(indata, y=targets)
            fitted = model.fit(indata, y=targets)
            components = np.dot(maps.T,indata)
        else:
            maps = model.fit_transform(indata)
            fitted = model.fit(indata)

            components = model.components_
        
        mapframe = pd.DataFrame(maps, index=inds)
        compframe = pd.DataFrame(components, columns=cols)

        temp_samp = mapframe.index.levels[0][0]

        temptList = []

        for samp in mapframe.index.levels[0]:
            tempframe = mapframe.unstack().xs(samp).unstack().T
            temptList.append(tempframe)

        newmaps = pd.concat(temptList, keys=(mapframe.index.levels[0]), axis=1)

        newmaps = newmaps.reindex(np.arange(0, self._gridsize[temp_samp] * self._gridsize[temp_samp]))

        results = analysis.Analysis(model=model, fitted=fitted, comps=compframe, maps=newmaps,
                                    gridSize=self._gridsize, samp_flags=samp_flags)

        self._analysis = results

        return newmaps, compframe