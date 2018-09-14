import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bepy import LineMeasurement, GridMeasurement
import os


class Sample:

    @property
    def linemeasurements(self):
        return

    @property
    def gridmeasurements(self):
        return
    
    @property
    def custom_line_flags(self):
        return
    
    @property
    def custom_pixel_flags(self):
        return

    def __init__(self, path=None, gridSize=50):

        self._linemeasurements = {}
        self._gridmeasurements = {}
        self._custom_line_flags = 0
        self._custom_pixel_flags = 0

        if path is not None:
            self.addentiresample(path, gridSize)

    def addmeasurement(self, meas, measType=None, gridSize=None):

        if measType is None:
            measType = meas._measurementName

            if measType == 'Scan':
                self._linemeasurements[measType] = meas
            elif measType in ['SSPFM', 'NonLin', 'Relax']:
                self._gridmeasurements[measType] = meas
            else:
                raise ValueError(measType+': Unknown measurement type')

        else:
            if measType == 'Scan':
                try:
                    newMeas = LineMeasurement(meas)
                except ValueError:
                    raise ValueError('Argument is not a path to measurement data. Maybe its a Measurement Object?')
                self._linemeasurements[measType] = newMeas
            elif measType in ['SSPFM', 'NonLin', 'Relax']:
                try:
                    newMeas = GridMeasurement(meas, measType, gridSize)
                except ValueError:
                    raise ValueError('Argument is not a path to measurement data. Maybe its a Measurement Object?')
                self._gridmeasurements[measType] = newMeas
            else:
                raise ValueError(measType+': Unknown measurement type')

    def addentiresample(self, path, gridSize=50):

        foldernames={'BELine': 'Scan', 'BEScan': 'Scan', 'Relax': 'Relax', 'SSPFM': 'SSPFM', 'NonLin': 'NonLin'}

        for name in foldernames:
            if os.path.isdir(path+name):
                self.addmeasurement(path+name+'/', measType=foldernames[name], gridSize=gridSize)


    def GetMeasStack(self, measstack = None, varstack = ['Amp', 'Phase', 'Res', 'Q'], inout=0.0, plotGroup=None, insert=None):

        if measstack is None:
            measstack = self._gridmeasurements.keys()

        stackedCols = 0
        stackreturn = 0

        for measure in measstack:

            measurement = self._gridmeasurements[measure]
            data = measurement.GetDataSubset(inout=inout, plotGroup=plotGroup, insert=insert, stack=varstack)
            
            cols = data.columns.values
            unzipped = np.asarray([list(t) for t in zip(*cols)])
            measname = np.repeat(measure,unzipped.shape[1])
            newcols = np.vstack([measname,unzipped])

            try:
                stackreturn = pd.concat([stackreturn, data], axis=1)
                stackedCols = np.hstack([stackedCols, newcols])
            except TypeError:
                stackreturn = data
                stackedCols = newcols
                
        tuples = list(zip(*stackedCols))

        cols = pd.MultiIndex.from_tuples(tuples, names=['Meas', 'Var', 'Chirp', 'InOut', 'PlotGroup', 'xaxis'])

        return pd.DataFrame(stackreturn.values, index=stackreturn.index, columns=cols)

    def plot(self, meas=None, variables=None, saveName=None, pointNum=None, InOut=0):

        if meas == 'Scan':
            try:
                measObj = self._linemeasurements[meas]
            except KeyError:
                raise KeyError('That measurement does not exist')
            measObj.plot(variables=variables, saveName=saveName)
        elif meas in ['SSPFM', 'NonLin', 'Relax']:
            try:
                measObj = self._gridmeasurements[meas]
            except KeyError:
                raise KeyError('That measurement does not exist')
            measObj.plot(variables=variables, pointNum=pointNum, InOut=InOut, saveName=saveName)
        else:
            raise ValueError('Please select which measurement to plot')