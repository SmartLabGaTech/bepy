import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bepy import LineMeasurement, GridMeasurement, Sample, Analysis, OtherFunctions
import sklearn.decomposition as learn
from scipy.signal import butter, lfilter, freqz

class SampleSet:

    @property
    def samples(self):
        return

    @property
    def analysis(self):
        return

    @property
    def gridsize(self):
        return


    def __init__(self, gridsize=None):
        self._samples = {}
        self._analysis = Analysis()

        if gridsize is None:
            self._gridsize = 0
        else:
            self._gridsize = gridsize

    def addsample(self, samp, sampName):
        self._samples[sampName] = samp

    def GetSampStack(self, sampstack=None, measstack=None, varstack=['Amp', 'Phase', 'Res', 'Q'], inout=0.0, plotGroup=None, insert=None, removeflagged=False):

        if sampstack is None:
            sampstack = self._samples.keys()

        newInd = 0
        stackreturn = 0

        for samp in sampstack:

            sample = self._samples[samp]

            if measstack is None:
                measstack = sample._gridmeasurements.keys()

            data = sample.GetMeasStack(measstack=measstack, varstack=varstack, inout=inout, plotGroup=plotGroup, insert=insert)
            
            #if removeflagged:
             #   data = data[sample.custom_line_flags]

            try:
                newInd = np.hstack([newInd, np.repeat(samp, data.shape[0])])
                stackreturn = pd.concat([stackreturn, data], axis=0)
            except TypeError:
                newInd = np.repeat(samp, data.shape[0])
                stackreturn = data
            except Exception as e:
                print(e)

        indicies = data.index.values
        oldInd = np.tile(indicies, len(sampstack))

        ind = np.vstack([newInd, oldInd])
        tuples = list(zip(*ind))

        index = pd.MultiIndex.from_tuples(tuples, names=['Sample', 'Point'])

        return pd.DataFrame(stackreturn.values, index=index, columns=stackreturn.columns)

    def fit(self, model, sampstack=None, measstack=None, varstack=['Amp', 'Phase', 'Res', 'Q'], inout=0.0, plotGroup=None, insert=None, removeflagged=False):

        data = self.GetSampStack(sampstack, measstack, varstack, inout, plotGroup, insert, removeflagged)

        indata = data.values[:]
        indata[np.where(indata==np.inf)] = 0
        indata[np.where(indata==np.nan)] = 0
        
        maps = model.fit_transform(indata)
        fitted = model.fit(indata)

        components = model.components_
        
        mapframe = pd.DataFrame(maps, index=data.index)
        compframe = pd.DataFrame(components, columns=data.columns)
        
        temptList=[]
        
        for samp in mapframe.index.levels[0]:
            temptList = mapframe.unstack().xs(samp).unstack().T
            test.append(temptList)
        
        newmaps = pd.concat(test,keys=(mapframe.index.levels[0]),axis=1)

        results = Analysis(model=model, fitted=fitted, comps=compframe, maps=newmaps, gridSize=self._gridsize)

        self._analysis = results

        return newmaps, compframe