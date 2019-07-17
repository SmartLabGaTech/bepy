import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import copy
from . import otherfunctions
from pathlib import Path

# Implement the data structure
class BaseMeasurement:

    # Store parameters of the measurement
    @property
    def params(self):
        return self._params

    # Store flags identifying bad chirps
    @property
    def flags(self):
        return self._flags

    # Store flags identifying bad acquisitions
    @property
    def acq_flags(self):
        return self._acq_flags

    # Store actual data
    @property
    def data(self):
        return self._data

    def __init__(self, shodata=None, parameters=None, xaxis=None, adjustphase=True):

        if shodata is None:
            self._data = 0
            self._params = 0
            self._flags = 0
            self._acq_flags = 0
        else:
            self._params = parameters.transpose()
            
            temp_plotgroup = shodata["PlotGroup"].xs(0)
            in_out = shodata['InOut'].unstack().xs(0)
            self._flags = shodata['Flag'].unstack()

            shodata['PR'] = np.zeros(shodata.shape[0])
            data = shodata[["Amp", "errA", "Phase", "errP", "Res", "errRes", "Q", "errQ", "PR"]].unstack()

            if adjustphase:
                temp = data['Phase'].replace([np.inf, -np.inf], np.nan).copy()

                phaseMean = temp.fillna(0).mean()
                phaseMean = phaseMean.replace([np.inf, -np.inf], np.nan)
                phaseMean = phaseMean.fillna(0).mean()

                data['Phase'] = data['Phase'] - phaseMean
                data['Phase'] = data['Phase'].applymap(lambda x: np.mod(x + np.pi, 2*np.pi) - np.pi)

            data['PR'] = data.apply(lambda row: row['Amp'] * np.sin(row['Phase']), axis=1)

            data = data.transpose()
            data['InOut'] = np.tile(in_out.values, 9)
            data.set_index('InOut', append=True, inplace=True)

            data['PlotGroup'] = np.tile(temp_plotgroup.values, 9)
            data.set_index('PlotGroup', append=True, inplace=True)
            
            if xaxis is not None:
                data['xaxis'] = np.tile(xaxis.values, 9)
                data.set_index('xaxis', append=True, inplace=True)
            
            data = data.transpose()

            self._data = data
            self.clean()

    def GetDataSubset(self, inout=0.0, plotGroup=None, insert=None, stack=None, clean=False):
        
        inout_vals=self._data.columns.get_level_values(level='InOut')
        plotGroup_vals=self._data.columns.get_level_values(level='PlotGroup')

        if stack is None:
            stack = ['Amp', 'Phase', 'Res', 'Q']

        if inout is None:
            inout_mask = np.ones(inout_vals.shape)
        else:
            inout_mask = inout_vals == inout
            
        if plotGroup is None:
            pg_mask = np.ones(plotGroup_vals.shape)
        else:
            pg_mask = plotGroup_vals == plotGroup
            
        mask = np.logical_and(inout_mask, pg_mask)

        if clean:
            cleanmask = self._acq_flags
        else:
            cleanmask = np.full(self._acq_flags.shape, False)

        return_data = copy.deepcopy(self._data)
        return_data = return_data[~cleanmask]

        if insert is None:
            return return_data.T[mask].T[stack]
        else:
            return_data.T[mask] = insert
            return return_data[stack]

    def clean(self, sensitivity=3, var=None, plot=False):

        if var is None:
            var = ['Amp', 'Phase', 'Q', 'Res', 'errA', 'errP', 'errQ', 'errRes']

        outflags = np.full(self._data[var].values.shape,False)

        mask = self._data[var].columns.get_level_values(level='InOut') ==0.0
        oodata = self._data[var].T[mask].T.values
        indata = self._data[var].T[~mask].T.values

        outflags[:, mask] = otherfunctions.cleanbychirp(oodata, sensitivity)
        outflags[:, ~mask] = otherfunctions.cleanbychirp(indata, sensitivity)

        if plot:
            plt.imshow(outflags, cmap='binary')
            plt.show()

        self._flags = pd.DataFrame(outflags, index=self._data[var].index, columns=self._data[var].columns)
        self._acq_flags = otherfunctions.collapseflags(self._flags)

        return self._acq_flags

    def export(self, inout=[0,1], plotgroups=[0], saveName=None):
        for i in inout:
            pg_data = []
            for pg in plotgroups:
                temp = self.GetDataSubset(inout=i, plotGroup=pg, stack=['Amp', 'Phase', 'Res', 'Q', 'errA', 'errP', 'errRes', 'errQ'])
                pg_data.append(temp)
            allData = pd.concat(pg_data, axis=1)

            if saveName is None:
                allData.to_csv('export_InOut_'+str(i)+'.csv')
            else:
                allData.to_csv(saveName+'_' + str(i) + '.csv')

# Implement Grid and Line measurements
class GridMeasurement(BaseMeasurement):

    _acqXaxis = None  # type: np.array()

    @property
    def gridSize(self):
        return self._gridSize
    
    @property
    def measurementName(self):
        return self._measurementName

    @property
    def acqXaxis(self):
        return self._acqXaxis

    def __init__(self, path=None, measType='SSPFM', gridSize=10, adjustphase=True):

        if type(path) is 'str':
            path = Path(path)

        shodata = pd.read_csv(path / 'shofit.csv', index_col=[0, 1])
        parameters = pd.read_csv(path / 'parameters.csv', header=None, index_col=0)

        if measType == 'SSPFM':
            self._acqXaxis = 10*(shodata['DC'].xs(0))
        elif measType == 'NonLin':
            self._acqXaxis = 10*(shodata['Multiplier'].xs(0))
        elif measType == 'Relax':
            self._acqXaxis = (shodata['Multiplier'].xs(0).index)*float(parameters.xs('Chirp Duration').values[0])
        
        BaseMeasurement.__init__(self, shodata, parameters, xaxis=self._acqXaxis, adjustphase=adjustphase)

        self._measurementName = measType
        self._gridSize = gridSize
        self.add_rc()

    def plot(self, variables=None, pointNum=None, InOut=0.0, insert=None, plotgroup=None, plotmap=False, saveName=None):
        if variables is None:
            variables = ['Amp', 'Phase', 'Res', 'Q']

        if type(pointNum) is tuple:
            pointNum = (pointNum[0] - 1) * self.gridSize + pointNum[1]

        subset = self.GetDataSubset(inout=InOut, plotGroup=plotgroup, insert=insert)
        num_vars = len(variables)
        rows = ((num_vars - 1) // 4) + 1
        if num_vars > 4:
            cols = 4
        else:
            cols = num_vars

        plt.figure(figsize=(5 * cols, 4 * rows))

        for i in np.arange(0, num_vars):
            var = variables[i]
            data = subset[var]
            xaxis = data.columns.get_level_values('xaxis')

            if var == 'Res':
                data = np.divide(data, 1000)
                ylabel = var + ' (kHz)'
            elif var == 'Amp':
                ylabel = var + r' ($\mu$V)'
            elif var == 'Phase':
                data = data * (180 / np.pi)
                ylabel = var + r' ($\degree$)'
            elif var == 'Q':
                ylabel = var

            if pointNum is None:
                plot_data = data.mean().values
            else:
                plot_data = data.xs(pointNum).values

            sub = plt.subplot(rows, cols, i + 1)

            if self.measurementName == 'SSPFM':
                sub.plot(xaxis, plot_data)
                sub.set_xlabel('DC Volt (V)')
            elif self.measurementName == 'NonLin':
                sub.plot(xaxis, plot_data)
                sub.set_xlabel('AC Volt (V)')
            elif self.measurementName == 'Relax':
                sub.plot(xaxis, plot_data)
                sub.set_xlabel('Time (s)')

            sub.set_ylabel(ylabel)

        if saveName is not None:
            plt.savefig(saveName)

        plt.show()

    def imslice(self, variables=None, sliceNum=None, InOut=0.0, plotgroup=None, saveName=None, limits=None):

        #none means standard variables
        if variables is None:
            variables = ['Amp', 'Phase', 'Res', 'Q']

        numVars = len(variables)

        #plot subset
        subset = self.GetDataSubset(inout=InOut, plotGroup=plotgroup)

        if numVars > 2:
            totalrows = 3
        else:
            totalrows = 2

        totalcols = int(np.floor((numVars+1)/2))

        for i in np.arange(0, numVars):

            var = variables[i]

            row = i % 2
            col = int(np.floor(i/2))

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

            plotdata = data.values.reshape((self._gridSize, self._gridSize, -1))

            sub = plt.subplot2grid((totalrows, totalcols), (row, col))

            if limits is not None:
                plot = sub.imshow(plotdata[:, :, sliceNum], cmap='nipy_spectral', vmin=limits[i][0], vmax=limits[i][1])
            else:
                plot = sub.imshow(plotdata[:, :, sliceNum], cmap='nipy_spectral')

            sub.set_title(ylabel)
            plt.colorbar(plot, ax=sub)

        sub = plt.subplot2grid((totalrows, totalcols), (totalrows-1, 0), colspan=totalcols)
        sub.plot(xaxis, 'b')
        sub.plot(sliceNum, xaxis[sliceNum], 'ro')
        sub.set_ylabel('Voltage')

        if saveName is not None:
            plt.savefig(saveName)

        plt.show()

    def movie(self, variables=None, InOut=0.0, plotgroup=None, saveName='movie.mp4', resolution=10, figsize=(15, 8)):

        metadata = dict(title=self.measurementName, artist='Matplotlib',
                        comment='')
        writer = FFMpegWriter(fps=15, metadata=metadata)

        fig = plt.figure(figsize=figsize)

        subset = self.GetDataSubset(inout=InOut, plotGroup=plotgroup)

        length = subset['Amp'].values.shape[1]

        limits = self.varlimits(variables=variables, InOut=InOut, plotgroup=plotgroup)

        with writer.saving(fig, saveName, 100):
            for k in np.arange(0, length, int(length/resolution)):
                # Create a new plot object
                self.imslice(variables=variables, sliceNum=k, InOut=InOut, plotgroup=plotgroup, limits=limits)
                writer.grab_frame()

    def varlimits(self, variables=None, InOut=0.0, plotgroup=None):

        # none means standard variables
        if variables is None:
            variables = ['Amp', 'Phase', 'Res', 'Q']

        numVars = len(variables)

        subset = self.GetDataSubset(inout=InOut, plotGroup=plotgroup)

        outdata = np.zeros((numVars, 2))

        for i in np.arange(0, numVars):

            var = variables[i]

            data = subset[var]

            data = data.replace([np.inf, -np.inf], np.nan)

            varmin = np.min(data.fillna(1000000).values)
            varmax = np.max(data.fillna(-1000000).values)

            if var == 'Res':
                varmin = np.divide(varmin, 1000)
                varmax = np.divide(varmax, 1000)
            elif var == 'Phase':
                varmin = 0 #varmin * (180/(np.pi))
                varmax = 360 #varmax * (180 / (np.pi))
            elif var == 'Q':
                varmax = 5000

            outdata[i][0] = varmin
            outdata[i][1] = varmax
        return outdata

    def add_rc(self):
        self._data['r'] = pd.Series(np.sort(np.tile(np.arange(self._gridSize), self._gridSize)), index=self._data.index)
        self._data['c'] = pd.Series(np.tile(np.arange(self._gridSize), self._gridSize), index=self._data.index)


class LineMeasurement(BaseMeasurement):

    @property
    def measurementName(self):
        return self._measurementName
    
    def __init__(self, path=None, name='Scan', adjustphase=True):

        if type(path) is 'str':
            path = Path(path)

        shodata = pd.read_csv(path / 'shofit.csv', index_col=[0, 1])
        parameters = pd.read_csv(path / 'parameters.csv', header=None, index_col=0)

        BaseMeasurement.__init__(self, shodata, parameters, xaxis=None, adjustphase=adjustphase)

        self._measurementName = name

    def plot(self, variables=None, fold=False, saveName=None, clean=False, plotgroup=None):
        if variables is None:
            variables = ['Amp', 'Phase', 'Res', 'Q']

        num_vars = len(variables)
        rows = ((num_vars - 1) // 4) + 1
        if num_vars > 4:
            cols = 4
        else:
            cols = num_vars

        plt.figure(figsize=(5 * cols, 4 * rows))

        for i in np.arange(0, num_vars):
            var = variables[i]
            data = self.GetDataSubset(stack=var, plotGroup=plotgroup)

            if clean:
                plot_data = copy.deepcopy(data.values)
                clean_flags = self._flags

                if plotgroup is not None:
                    pg_mask = self._data.columns.get_level_values(level='PlotGroup') == plotgroup

                clean_flags = clean_flags.T[pg_mask].T
                plot_data[clean_flags[var].values] = np.inf
            else:
                plot_data = data.values

            if fold:
                image_rows = np.shape(plot_data)[0]
                image_cols = np.shape(plot_data)[1]
                top = plot_data[0:int(image_rows / 2), :]
                bottom = np.flipud(plot_data[int(image_rows / 2):, :])
                new_plot_data = np.empty([int(image_rows), image_cols])
                new_plot_data[::2, :] = top
                new_plot_data[1::2, :] = bottom
                plot_data = new_plot_data

            if var == 'Res':
                plot_data = np.divide(plot_data, 1000000)

            if var == 'Phase':
                plot_data = np.multiply(plot_data, 180 / np.pi)

            sub = plt.subplot(rows, cols, i + 1)
            minimum = np.mean(plot_data) - 1.5 * np.std(plot_data)
            maximum = np.mean(plot_data) + 1.5 * np.std(plot_data)
            plot = sub.imshow(plot_data, cmap='viridis', vmin=minimum, vmax=maximum)
            plt.colorbar(plot, ax=sub)
            sub.set_title(var)

        if saveName is not None:
            plt.savefig(saveName)

        plt.show()
