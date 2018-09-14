import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#implement the data structure
from typing import Any, Union


class BEData:

    @property
    def amp(self):
        return

    @property
    def errAmp(self):
        return

    @property
    def phase(self):
        return

    @property
    def errPhase(self):
        return

    @property
    def resonance(self):
        return

    @property
    def errRes(self):
        return

    @property
    def quality(self):
        return

    @property
    def errQ(self):
        return

    @property
    def params(self):
        return

    @property
    def inoutmask(self):
        return

    @property
    def plotgroupmask(self):
        return

    def __init__(self, shodata=None, parameters=None):

        if shodata is None:

            self._amp = 0
            self._errAmp = 0

            self._phase = 0
            self._errPhase = 0

            self._resonance = 0
            self._errRes = 0

            self._quality = 0
            self._errQ = 0

            self._params = 0

            self._inoutmask = 0
            self._plotgroupmask = 0

        else:

            self._amp = shodata["Amp"].unstack()
            self._errAmp = shodata["errA"].unstack()

            self._phase = shodata["Phase"].unstack()
            self._errPhase = shodata["errP"].unstack()

            self._resonance = shodata["Res"].unstack()
            self._errRes = shodata["errRes"].unstack()

            self._quality = shodata["Q"].unstack()
            self._errQ = shodata["errQ"].unstack()

            self._params = parameters.transpose()

            self._inoutmask = shodata["InOut"].xs(0) == 0

            temp_plotgroup = shodata["PlotGroup"].xs(0)
            max_plotgroup = np.max(temp_plotgroup)
            masks = []

            for i in np.arange(0, max_plotgroup):

                mask = temp_plotgroup == i
                masks.append(mask)

            self._plotgroupmask = masks

    def GetVarStack(self, regOrError = 'reg', stack = None, axis = 1):

        if stack is None:
            stack = ['amp', 'phase', 'res', 'Q']

        ind = 0
        stackreturn = 0

        if regOrError.lower() == 'reg':

            for variable in stack:

                if variable.lower() == 'amp':
                    data = self._amp
                elif variable.lower() == 'phase':
                    data = self._phase
                elif variable.lower() == 'res':
                    data = self._resonance
                elif variable.lower() == 'q':
                    data = self._quality
                else:
                    raise ValueError(variable+': not a valid argument for stack')

                try:
                    ind = np.hstack([ind, np.repeat(variable, data.shape[1])])
                    stackreturn = pd.concat([stackreturn, data], axis = axis)
                except TypeError:
                    ind = np.repeat(variable, data.shape[1])
                    stackreturn = data

        elif regOrError.lower() == 'error':

            for variable in stack:

                if variable.lower() == 'amp':
                    data = self._errAmp
                elif variable.lower() == 'phase':
                    data = self._errPhase
                elif variable.lower() == 'res':
                    data = self._errRes
                elif variable.lower() == 'Q':
                    data = self._errQ
                else:
                    raise ValueError(variable+': not a valid argument for stack')

                try:
                    ind = np.hstack([ind, np.repeat(variable, data.shape[1])])
                    stackreturn = pd.concat([stackreturn, data], axis = axis)
                except TypeError:
                    ind = np.repeat(variable, data.shape[1])
                    stackreturn = data

        else:
            raise ValueError('regOrError should be "reg" or "error", representing the types of Value to return')

        stackreturn.columns = ind

        return stackreturn


#Implement tha base measurement class
class BaseMeasurement(BEData):
    pass


#Implement Grid and Line measurements
class GridMeasurement(BaseMeasurement):

    _acqXaxis = None  # type: np.array()

    @property
    def measurementName(self):
        return

    @property
    def gridSize(self):
        return

    @property
    def acqXaxis(self):
        return

    def __init__(self, path=None, measType='SSPFM', gridSize=10):

        shodata = pd.read_csv(path + 'shofit.csv', index_col=[0, 1])
        parameters = pd.read_csv(path + 'parameters.csv', header=None, index_col=0)

        BaseMeasurement.__init__(self, shodata, parameters)

        self._measurementName = measType
        self._gridSize = gridSize

        if measType == 'SSPFM':
            self._acqXaxis = 10*(shodata['DC'].xs(0))
        elif measType == 'NonLin':
            self._acqXaxis = 10*(shodata['Multiplier'].xs(0))
        elif measType == 'Relax':
            self._acqXaxis = (self._amp.columns-1)*float(self._params['Chirp Duration'].values[0])

        self._amp.columns = self._acqXaxis
        self._errAmp.columns = self._acqXaxis

        self._phase.columns = self._acqXaxis
        self._errPhase.columns = self._acqXaxis

        self._resonance.columns = self._acqXaxis
        self._errRes.columns = self._acqXaxis

        self._quality.columns = self._acqXaxis
        self._errQ.columns = self._acqXaxis

    def plot(self, variables=None, regOrError='reg', pointNum=None, InOut=0, saveName=None):

        if variables is None:
            variables = ['Amp', 'Phase', 'Res', 'Q']

        numVars=len(variables)

        rows = (numVars // 3) + 1

        if rows < numVars:
            cols = 2
        else:
            cols = 1

        for i in np.arange(0, numVars):

            var = variables[i]

            if regOrError.lower() == 'reg':

                if var.lower() == 'amp':
                    data = self._amp
                elif var.lower() == 'phase':
                    data = self._phase
                elif var.lower() == 'res':
                    data = self._resonance
                elif var.lower() == 'q':
                    data = self._quality
                else:
                    raise ValueError(var+': not a valid argument for stack')

            elif regOrError.lower() == 'error':

                if var.lower() == 'amp':
                    data = self._errAmp
                elif var.lower() == 'phase':
                    data = self._errPhase
                elif var.lower() == 'res':
                    data = self._errRes
                elif var.lower() == 'q':
                    data = self._errQ
                else:
                    raise ValueError(var+': not a valid argument for stack')

            else:
                raise ValueError('regOrError should be "reg" or "error", representing the types of Value to return')

            if pointNum is None:
                plotdata = data.mean().values
            else:
                plotdata = data.xs(pointNum).values

            if InOut in [0, 1]:
                mask = (self._inoutmask ^ InOut)
            else:
                mask = np.repeat(True, len(self._inoutmask))

            plotdata[mask] = plotdata[mask]
            plotdata[~mask] = np.inf

            sub = plt.subplot(rows, cols, i + 1)
            sub.plot(self._acqXaxis, plotdata)

            sub.set_ylabel(var)

            if self._measurementName == 'SSPFM':
                sub.set_xlabel('DC Volt (V)')
            elif self._measurementName == 'NonLin':
                sub.set_xlabel('AC Volt (V)')
            elif self._measurementName == 'Relax':
                sub.set_xlabel('Time (s)')

        if saveName is not None:
            plt.savefig(saveName)

        plt.show()


class LineMeasurement(BaseMeasurement):

    def __init__(self, path = None):

        shodata = pd.read_csv(path + 'shofit.csv', index_col=[0, 1])
        parameters = pd.read_csv(path + 'parameters.csv', header=None, index_col=0)

        BaseMeasurement.__init__(self, shodata, parameters)

        # Some meaningless default parameters
        if path is None:
            self.acqType = 'Line'
            self.measurementName = 'Scan'

    def plot(self, variables=None, regOrError='reg', saveName=None):

        if variables is None:
            variables = ['Amp', 'Phase', 'Res', 'Q']

        numVars = len(variables)

        rows = (numVars // 3) + 1

        if rows < numVars:
            cols = 2
        else:
            cols = 1

        for i in np.arange(0, numVars):

            var = variables[i]

            if regOrError.lower() == 'reg':

                if var.lower() == 'amp':
                    data = self._amp
                elif var.lower() == 'phase':
                    data = self._phase
                elif var.lower() == 'res':
                    data = self._resonance
                elif var.lower() == 'q':
                    data = self._quality
                else:
                    raise ValueError(var+': not a valid argument for stack')

            elif regOrError.lower() == 'error':

                if var.lower() == 'amp':
                    data = self._errAmp
                elif var.lower() == 'phase':
                    data = self._errPhase
                elif var.lower() == 'res':
                    data = self._errRes
                elif var.lower() == 'Q':
                    data = self._errQ
                else:
                    raise ValueError(var+': not a valid argument for stack')

            else:
                raise ValueError('regOrError should be "reg" or "error", representing the types of Value to return')

            plotdata = data.values

            sub = plt.subplot(rows, cols, i + 1)
            plot = sub.imshow(plotdata, cmap='nipy_spectral')

            plt.colorbar(plot, ax=sub)

            sub.set_title(var)

        if saveName is not None:
            plt.savefig(saveName)

        plt.show()


