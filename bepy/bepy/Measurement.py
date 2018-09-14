import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
from typing import Any, Union

#implement the data structure

class BEData:

    @property
    def params(self):
        return

    @property
    def flags(self):
        return
    
    @property
    def data(self):
        return

    def __init__(self, shodata=None, parameters=None, xaxis=None):

        if shodata is None:
            self._data = 0
            self._params = 0
            self._flags = 0
        else:
            data = shodata[["Amp","errA","Phase","errP","Res","errRes","Q","errQ"]].unstack()
            self._params = parameters.transpose()
            
            temp_plotgroup = shodata["PlotGroup"].xs(0)
            in_out = shodata['InOut'].unstack().xs(0)
            self._flags = shodata['Flag'].unstack()
            
            data = shodata[["Amp","errA","Phase","errP","Res","errRes","Q","errQ"]].unstack()

            data = data.transpose()
            data['InOut']= np.tile(in_out.values,8)
            data.set_index('InOut', append=True, inplace=True)

            data['PlotGroup']= np.tile(temp_plotgroup.values,8)
            data.set_index('PlotGroup', append=True, inplace=True)
            
            if xaxis is not None:
                data['xaxis']= np.tile(xaxis.values,8)
                data.set_index('xaxis', append=True, inplace=True)
            
            data = data.transpose()
            
            self._data = data
    
    def GetDataSubset(self, inout=0.0, plotGroup=None, insert=None, stack=['Amp', 'Phase', 'Res', 'Q']):
        
        inout_vals=self._data.columns.get_level_values(level='InOut')
        plotGroup_vals=self._data.columns.get_level_values(level='PlotGroup')
        
        if inout is None:
            inout_mask = np.ones(inout_vals.shape)
        else:
            inout_mask = inout_vals == inout
            
        if plotGroup is None:
            pg_mask = np.ones(plotGroup_vals.shape)
        else:
            pg_mask = plotGroup_vals == plotGroup
            
        mask = np.logical_and(inout_mask,pg_mask)
        
        if insert is None:
            return self._data.T[mask].T[stack]
        else:
            return_data = copy.deepcopy(self._data)
            return_data.T[mask] = insert
            return return_data[stack]
    

#Implement tha base measurement class
class BaseMeasurement(BEData):
    pass


#Implement Grid and Line measurements
class GridMeasurement(BaseMeasurement):

    _acqXaxis = None  # type: np.array()

    @property
    def gridSize(self):
        return
    
    @property
    def measurementName(self):
        return

    @property
    def acqXaxis(self):
        return

    def __init__(self, path=None, measType='SSPFM', gridSize=10):

        shodata = pd.read_csv(path + 'shofit.csv', index_col=[0, 1])
        parameters = pd.read_csv(path + 'parameters.csv', header=None, index_col=0)

        if measType == 'SSPFM':
            self._acqXaxis = 10*(shodata['DC'].xs(0))
        elif measType == 'NonLin':
            self._acqXaxis = 10*(shodata['Multiplier'].xs(0))
        elif measType == 'Relax':
            self._acqXaxis = (shodata['Multiplier'].xs(0).index)*float(parameters.xs('Chirp Duration').values[0])
        
        BaseMeasurement.__init__(self, shodata, parameters, xaxis=self._acqXaxis)

        self._measurementName = measType
        self._gridSize = gridSize

    def plot(self, variables=None, pointNum=None, InOut=0.0, insert=None, plotgroup=None, saveName=None):

        if variables is None:
            variables = ['Amp', 'Phase', 'Res', 'Q']

        subset = self.GetDataSubset(inout=InOut, plotGroup=plotgroup, insert=insert)
        
        numVars=len(variables)

        rows = ((numVars-1) // 4) + 1

        if numVars > 4:
            cols = 4
        else:
            cols = numVars

        for i in np.arange(0, numVars):

            var = variables[i]

            data = subset[var]
            xaxis = data.columns.get_level_values('xaxis')

            if var == 'Res':
                data = np.divide(data, 1000)
                ylabel = var + ' (kHz)'
            elif var == 'Amp':
                ylabel = var + r' ($\mu$V)'
            elif var == 'Phase':
                data = data * (180/(np.pi))
                ylabel = var + r' ($\degree$)'
            elif var == 'Q':
                ylabel = var

            if pointNum is None:
                plotdata = data.mean().values
            else:
                plotdata = data.xs(pointNum).values

            sub = plt.subplot(rows, cols, i + 1)

            if self._measurementName == 'SSPFM':
                sub.plot(xaxis, plotdata)
                sub.set_xlabel('DC Volt (V)')
            elif self._measurementName == 'NonLin':
                sub.plot(xaxis, plotdata)
                sub.set_xlabel('AC Volt (V)')
            elif self._measurementName == 'Relax':
                sub.plot(xaxis, plotdata)
                sub.set_xlabel('Time (s)')

            sub.set_ylabel(ylabel)

        if saveName is not None:
            plt.savefig(saveName)

        plt.show()


class LineMeasurement(BaseMeasurement):

    @property
    def measurementName(self):
        return
    
    def __init__(self, path = None):

        shodata = pd.read_csv(path + 'shofit.csv', index_col=[0, 1])
        parameters = pd.read_csv(path + 'parameters.csv', header=None, index_col=0)

        BaseMeasurement.__init__(self, shodata, parameters, xaxis=None)

        self._measurementName = 'Scan'

    def plot(self, variables=None, removeFlagged = False, fold = False, lims = None, saveName=None):

        if variables is None:
            variables = ['Amp', 'Phase', 'Res', 'Q']

        numVars = len(variables)

        rows = ((numVars - 1) // 4) + 1

        if numVars > 4:
            cols = 4
        else:
            cols = numVars

        for i in np.arange(0, numVars):

            var = variables[i]

            data = self._data[var]

            plotdata = data.values

            if fold:
                imRows = np.shape(plotdata)[0]
                imCols = np.shape(plotdata)[1]
                top = plotdata[0:int(imRows/2),:]
                bottom = np.flipud(plotdata[int(imRows/2):,:])
                new_plotdata = np.empty([int(imRows), imCols])
                new_plotdata[::2,:] = top
                new_plotdata[1::2,:] = bottom
                plotdata = new_plotdata

            if removeFlagged:
                plotdata[self._flags.values.astype(bool)] = np.inf
                
            if lims is None:
                minimum = np.min(plotdata)
                maximum = np.max(plotdata)
            else:
                minimum = lims[0][i]
                maximum = lims[1][i]

            if var == 'Res':
                plotdata = np.divide(plotdata, 1000000)

            sub = plt.subplot(rows, cols, i + 1)
            plot = sub.imshow(plotdata, cmap='nipy_spectral', vmin=minimum, vmax=maximum)

            plt.colorbar(plot, ax=sub)

            sub.set_title(var)

        if saveName is not None:
            plt.savefig(saveName)

        plt.show()
        
        

