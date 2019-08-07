import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import copy
from . import otherfunctions
from pathlib import Path
from scipy import stats, ndimage, integrate
import warnings
import os
from skimage import feature
import statistics


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

    @property
    def path(self):
        return self._path

    @property
    def analysis(self):
        return self._analysis

    def __init__(self, path=None, measType='SSPFM', gridSize=10, adjustphase=True):

        if type(path) is 'str':
            path = Path(path)
        self._path = path

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
        if os.path.isfile(self.path / 'analysis.csv'):
            self._analysis = pd.read_csv(os.path.join(self.path, 'analysis.csv'))
            self._analysis = self._analysis.set_index('Acq')
        else:
            self._analysis = pd.DataFrame(index=self.data.index)

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

    @staticmethod
    def intersect_lines(m1, b1, m2, b2):
        x = (b2 - b1) / (m1 - m2)
        y = m1 * x + b1
        return x, y

    def split_loop(self, index, second_loop=False, inout=0, stack='PR', down_up=True):
        full_v = self.GetDataSubset(inout=inout, stack=stack).columns.get_level_values(3).values
        full_pr = self.GetDataSubset(inout=inout, stack=stack).iloc[index].values
        middle_index = int(len(full_pr) / 2)
        if second_loop:
            half_v = full_v[middle_index:-1]  # Out-of-field loops have an extra point at the end.
            half_pr = full_pr[middle_index:-1]  # if this is changed, the -1 will need to be removed...
        else:
            half_v = full_v[:middle_index]
            half_pr = full_pr[:middle_index]

        # Triangular down up
        if down_up:
            min_idx = np.argmin(half_v)
            max_idx = np.argmax(half_v)
            branch_v = half_v[min_idx:max_idx + 1]
            branch_pr_1 = half_pr[min_idx:max_idx + 1]
            branch_pr_2 = np.concatenate((half_pr[:min_idx + 1][::-1], half_pr[max_idx:][::-1]))
            if np.mean(branch_pr_1) > np.mean(branch_pr_2):
                return branch_v.astype(float), branch_pr_1.astype(float), branch_pr_2.astype(float)
            else:
                return branch_v.astype(float), branch_pr_2.astype(float), branch_pr_1.astype(float)

        # Triangular up down
        else:
            min_idx = np.argmin(half_v)
            max_idx = np.argmax(half_v)
            branch_v = half_v[max_idx:min_idx + 1]
            branch_pr_1 = half_pr[max_idx:min_idx + 1]
            branch_pr_2 = np.concatenate((half_pr[:max_idx + 1][::-1], half_pr[min_idx:][::-1]))
            if np.mean(branch_pr_1) > np.mean(branch_pr_2):
                return branch_v, branch_pr_1, branch_pr_2
            else:
                return branch_v, branch_pr_2, branch_pr_1

    def extract_sspfm_parameters(self, index, horz_spread=20, vert_spread=2, second_loop=False, inout=0, stack='PR',
                                 plot_ax=None, show_legend=False):
        v, pr_up, pr_down = self.split_loop(index, second_loop, inout, stack)
        try:
            v_diff = np.diff(v)

            pr_up_diff = np.diff(pr_up)
            pr_up_slopes = np.add(pr_up_diff[1:], pr_up_diff[:-1]) / np.add(v_diff[1:], v_diff[:-1])
            pr_up_slopes_filtered = abs(ndimage.filters.gaussian_filter(list(pr_up_slopes), 1, mode='reflect'))
            pr_up_max_slopes_filtered_idx = np.argmax(pr_up_slopes_filtered)

            pr_down_diff = np.diff(pr_down)
            pr_down_slopes = np.add(pr_down_diff[1:], pr_down_diff[:-1]) / np.add(v_diff[1:], v_diff[:-1])
            pr_down_slopes_filtered = abs(ndimage.filters.gaussian_filter(list(pr_down_slopes), 1, mode='reflect'))
            pr_down_max_slopes_filtered_idx = np.argmax(pr_down_slopes_filtered)

            # pr_up and pr_down have the same first and last point
            max_v_idx = np.argmax(v)
            min_v_idx = np.argmin(v)
            v_sat_pos = v[max_v_idx]
            pr_sat_pos = pr_up[max_v_idx]
            v_sat_neg = v[min_v_idx]
            pr_sat_neg = pr_up[min_v_idx]

            vert_left_m, vert_left_b, _, _, _ = stats.linregress(
                v[pr_up_max_slopes_filtered_idx - vert_spread:pr_up_max_slopes_filtered_idx + vert_spread + 1],
                pr_up[pr_up_max_slopes_filtered_idx - vert_spread:pr_up_max_slopes_filtered_idx + vert_spread + 1])
            vert_right_m, vert_right_b, _, _, _ = stats.linregress(v[pr_down_max_slopes_filtered_idx - vert_spread:
                                                                     pr_down_max_slopes_filtered_idx + vert_spread + 1]
                                                                   [::-1],
                pr_down[pr_down_max_slopes_filtered_idx - vert_spread:
                       pr_down_max_slopes_filtered_idx + vert_spread + 1][::-1])
            v_c_pos = -vert_right_b / vert_right_m
            v_c_neg = -vert_left_b / vert_left_m

            min_abs_v = np.argmin(abs(v))
            horz_upper_m, horz_upper_b, _, _, _ = stats.linregress(v[-horz_spread:], pr_up[-horz_spread:])
            horz_lower_m, horz_lower_b, _, _, _ = stats.linregress(v[:horz_spread], pr_down[:horz_spread])
            v_nuc_pos, pr_nuc_pos = self.intersect_lines(horz_lower_m, horz_lower_b, vert_right_m, vert_right_b)
            v_nuc_neg, pr_nuc_neg = self.intersect_lines(horz_upper_m, horz_upper_b, vert_left_m, vert_left_b)

            if second_loop:
                _, pr_rem_pos, _, _, _ = stats.linregress(v[min_abs_v - 2:min_abs_v + 4], pr_up[min_abs_v - 2:min_abs_v + 4])
                _, pr_rem_neg, _, _, _ = stats.linregress(v[min_abs_v - 2:min_abs_v + 4], pr_down[min_abs_v - 2:min_abs_v + 4])
            else:
                # assumes down-up triangular waveform
                pr_rem_pos = pr_up[min_abs_v+1]
                pr_rem_neg = pr_down[min_abs_v]

            pr_s = pr_sat_pos - pr_sat_neg
            imprint = 0.5 * (v_c_pos + v_c_neg)
            area = integrate.trapz(pr_up - pr_down, x=v)

            if plot_ax is not None:
                plot_ax.plot(v, pr_up, '-k')
                plot_ax.plot(v, pr_down, '-k')
                plot_ax.set_xlabel('Voltage (V)')
                plot_ax.set_ylabel('PR (a.u.)')

                x_left = v[pr_up_max_slopes_filtered_idx-15:pr_up_max_slopes_filtered_idx+16]
                x_right = v[pr_down_max_slopes_filtered_idx-15:pr_down_max_slopes_filtered_idx+16]
                plot_ax.plot(x_left, vert_left_m * x_left + vert_left_b, '--r')
                plot_ax.plot(x_right, vert_right_m * x_right + vert_right_b, '--r')

                # x_up = np.concatenate((voltage[min_abs_v[0] - 30:], voltage[:min_abs_v[0] + 30]))
                # x_down = voltage[min_abs_v[2] - 30:min_abs_v[2] + 30]
                x = np.array([min(v), max(v)])
                plot_ax.plot(x, horz_upper_m * x + horz_upper_b, '--r')
                plot_ax.plot(x, horz_lower_m * x + horz_lower_b, '--r')

                line1, = plot_ax.plot([0, 0], [pr_rem_pos, pr_rem_neg], 'k*')
                line2, = plot_ax.plot([v_sat_pos, v_sat_neg], [pr_sat_pos, pr_sat_neg], 'ko')
                line3, = plot_ax.plot([v_c_pos, v_c_neg], [0, 0], 'ks')
                line4, = plot_ax.plot([v_nuc_pos, v_nuc_neg], [pr_nuc_pos, pr_nuc_neg], 'kd')
                plot_ax.set_ylim(np.min(pr_up) * 1.2, np.max(pr_up) * 1.2)

                if show_legend:
                    plot_ax.legend((line1, line2, line3, line4), ('Remanent', 'Saturated', 'Coercive', 'Nucleation'),
                                   loc='lower right')

            return {'Area': area, 'PRrem+': pr_rem_pos, 'PRrem-': pr_rem_neg, 'PRsat+': pr_sat_pos,
                    'PRsat-': pr_sat_neg, 'Vc+': v_c_pos, 'Vc-': v_c_neg, 'Vnuc+': v_nuc_pos, 'Vnuc-': v_nuc_neg,
                    'Imprint': imprint, 'PRs': pr_s}

        except Exception as exc:
            print(exc)
            return {'Area': np.nan, 'PRrem+': np.nan, 'PRrem-': np.nan, 'PRsat+': np.nan, 'PRsat-': np.nan,
                    'Vc+': np.nan, 'Vc-': np.nan, 'Vnuc+': np.nan, 'Vnuc-': np.nan, 'Imprint': np.nan, 'PRs': np.nan}

    def extract_all_sspfm_parameters(self, horz_spread=20, vert_spread=2, second_loop=False):
        num_rows = len(self.data.index)
        area = np.zeros(num_rows)
        pr_rem_pos = np.zeros(num_rows)
        pr_rem_neg = np.zeros(num_rows)
        pr_sat_pos = np.zeros(num_rows)
        pr_sat_neg = np.zeros(num_rows)
        v_c_pos = np.zeros(num_rows)
        v_c_neg = np.zeros(num_rows)
        v_nuc_pos = np.zeros(num_rows)
        v_nuc_neg = np.zeros(num_rows)
        imprint = np.zeros(num_rows)
        pr_s = np.zeros(num_rows)
        for index in range(num_rows):
            result = self.extract_sspfm_parameters(index, horz_spread=horz_spread, vert_spread=vert_spread,
                                                   second_loop=second_loop)
            area[index] = result['Area']
            pr_rem_pos[index] = result['PRrem+']
            pr_rem_neg[index] = result['PRrem-']
            pr_sat_pos[index] = result['PRsat+']
            pr_sat_neg[index] = result['PRsat-']
            v_c_pos[index] = result['Vc+']
            v_c_neg[index] = result['Vc-']
            v_nuc_pos[index] = result['Vnuc+']
            v_nuc_neg[index] = result['Vnuc-']
            imprint[index] = result['Imprint']
            pr_s[index] = result['PRs']
            if np.mod(index, 100) == 0:
                print('Progress: ' + str(index) + '/' + str(num_rows))
        self.analysis['Area'] = area
        self.analysis['PRrem+'] = pr_rem_pos
        self.analysis['PRrem-'] = pr_rem_neg
        self.analysis['PRsat+'] = pr_sat_pos
        self.analysis['PRsat-'] = pr_sat_neg
        self.analysis['Vc+'] = v_c_pos
        self.analysis['Vc-'] = v_c_neg
        self.analysis['Vnuc+'] = v_nuc_pos
        self.analysis['Vnuc-'] = v_nuc_neg
        self.analysis['Imprint'] = imprint
        self.analysis['PRs'] = pr_s

    def plot_loop(self, index, plot_ax, second_loop=False, inout=0, stack='PR'):
        v, pr_up, pr_down = self.split_loop(index, second_loop, inout, stack)
        plot_ax.plot(v, pr_up, '-k')
        plot_ax.plot(v, pr_down, '-k')
        plot_ax.set_xlabel('Voltage (V)')
        plot_ax.set_ylabel(stack)

    def save_analysis(self):
        if self.path is not None:
            self.analysis.to_csv(os.path.join(self.path, 'analysis.csv'))
        else:
            warnings.warn('Path not set. Set the path to the GridMeasurement files using the .path property to this ' +
                          'measurement object.')

    def load_analysis(self):
        if self.path is not None:
            if os.path.isfile(self.path / 'analysis.csv'):
                self._analysis = pd.read_csv(os.path.join(self.path, 'SSPFM_analysis.csv'))
                self._analysis = self._analysis.set_index('Acq')
            else:
                warnings.warn('Saved SSPFM analysis file does not exist.')
        else:
            warnings.warn('Path not set. Set the path to the GridMeasurement files using the .path property to this ' +
                          'measurement object.')


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
            img = sub.imshow(plot_data, cmap='inferno', vmin=minimum, vmax=maximum)
            plt.colorbar(img, ax=sub)
            sub.set_title(var)

        if saveName is not None:
            plt.savefig(saveName)

        plt.show()

    def detect_domain_walls(self, sigma=3, stack='Phase'):
        return feature.canny(self.GetDataSubset(plotGroup=1, stack=stack).values, sigma)

    def find_distances(self, domain_walls=None):
        if domain_walls is None:
            domain_walls = self.detect_domain_walls()
        num_rows = domain_walls.shape[0]
        num_cols = domain_walls.shape[1]
        distances = np.empty((num_rows, num_cols))
        master = otherfunctions.generate_distance_kernel((num_rows, num_cols))
        for row in range(num_rows):
            for col in range(num_cols):
                sub_master = master[num_rows - 1 - row:2 * num_rows - 1 - row,
                                    num_cols - 1 - col:2 * num_cols - 1 - col]
                distances[row, col] = np.min(sub_master[domain_walls])
        return distances
