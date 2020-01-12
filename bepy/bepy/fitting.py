from nptdms import TdmsFile  # For reading LabView files
import numpy as np  # for numerical operations
import pandas as pd  # for data reading/structure
import matplotlib.pyplot as plt  # Used for plotting
from scipy.optimize import curve_fit
import scipy.signal as signal
from lmfit import Model, Parameters, minimize
import lmfit as lm
from functools import partial
import sys


# Pass the frequency spectrum for a chirp and the frequnecy axis, and this function will return a guess for the
# SHO fit
def getSHOguess(chirpData, freq):
    # Get amplitude and phase
    amp = np.abs(chirpData)
    phase = np.angle(chirpData)

    # -----------------Get some guesses for the fitting-----------------
    resGuess = np.argmax(amp)
    res = freq[resGuess]
    ampGuess = amp[resGuess]

    # Start by estimating the full width half max
    ampHalfMax = ampGuess / 2

    # First find the inidices (left and right) where the value is half the maximum
    # An error is thrown is these indicies or outside the range of the data
    try:
        leftFW = np.where(amp[0:resGuess] > ampHalfMax)
        leftInd = leftFW[0][0]
    except IndexError:
        leftInd = 0

    try:
        rightFW = np.where(amp[resGuess:] < ampHalfMax)
        rightInd = rightFW[0][0] + resGuess
    except:
        rightInd = len(amp) - 1

    FWHM = freq[rightInd] - freq[leftInd]

    temp = np.unwrap(phase)
    phi = np.mod(temp[resGuess] + 3 * np.pi / 2, 2 * np.pi)
    Q = np.abs(res / FWHM)

    if Q > 1000:
        Q = 500

    a = ampGuess / Q

    xGuess = [a, phi, res, Q]

    return xGuess


# Complex gaussian function describing a damped-driven oscillator
def complexGaus(x, a, phi, res, Q):
    func = a * np.exp(1j * phi) * res ** 2 / (x ** 2 - 1j * x * res / Q - res ** 2)
    return func


# Amplitude of the complex gaussian
def complexGausAmp(x, a, res, Q):
    func = (a * res ** 2) / (np.sqrt(((x ** 2 - res ** 2) ** 2) + ((x ** 2 * res ** 2) / (Q ** 2))))
    return func


# A standard moving average filter for data smoothing
def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


# Function that is used for fitting. It calucaltes the complex gaussian and returns the error for given data
def fitFunc(pars, x, data=None):
    vals = pars.valuesdict()
    a = vals['a']
    phi = vals['phi']
    res = vals['res']
    Q = vals['Q']
    # offset=vals['Offset']

    fit = complexGaus(x, a, phi, res, Q)
    realRes = np.real(fit)  # +offset
    imagRes = np.imag(fit)  # +offset
    combData = np.concatenate((realRes, imagRes))

    if data is None:
        return combData

    return combData - data


# Alternative Function used for fitting, instead it calculates just the amplitude of the complex gaussian
def fitFuncAmp(pars, x, data=None):
    vals = pars.valuesdict()
    a = vals['a']
    # phi=vals['phi']
    res = vals['res']
    Q = vals['Q']
    # offset=vals['Offset']

    fit = complexGausAmp(x, a, res, Q)
    # amp=np.abs(fit)
    combData = fit  # +offset

    if data is None:
        return combData

    return combData - data


# This function smooths the data prior to fitting, if requested
def smoothChirp(chirpData, freq):
    # Automatically choosing the degree of something based off of the number of points
    n = int(len(freq) / 30)

    # Smooth
    smoothed_real = moving_average(np.real(chirpData), n=n)
    smoothed_imag = moving_average(np.imag(chirpData), n=n)
    freq = moving_average(freq, n=n)

    smoothed = smoothed_real + 1j * smoothed_imag

    return smoothed, freq


# This funcion "detrends" the data prior to fitting, if requested. I.e. it removes a linear background
def remove_linear(chirpData):
    return signal.detrend(chirpData)


# This function tries to trim the data according to the user-provided low and high frequency limits
def attempt_trim(chirpData, freq, lowFreq, highFreq):
    # Try to trim the data according to the provided freqeuncy limits.
    # If it fails recore the exception
    try:
        # Trim unecessary data
        if lowFreq is None:
            trimData, trimFreq = limFitData(chirpData, freq, lowFreq, highFreq)
        else:
            trimData = chirpData
            trimFreq = freq

        freqData = trimFreq
        failed = False

    except Exception as e:
        freqData = freq
        failed = True

    return trimData, freqData, failed


# Perform the fit, initalize parameters, choose which data is actually fit, run it.
# You can fit the: Real and imaginary data ('RealImag') or the Amplitude data ('Amp')
def performFit(chirpData, freqData, guesses, fitType, acqNum, chirpNum, print_report=False):
    if fitType is 'Amp':
        fitdata, params, phase = setupAmpFit(chirpData, freqData, guesses)
        pfit, perr, failed, fail_msg, iterations = performAmpFit(fitdata, freqData, params, phase, acqNum, chirpNum,
                                                                 print_report)
    elif fitType is 'RealImag':
        fitdata, params = setupRealFit(chirpData, freqData, guesses)
        pfit, perr, failed, fail_msg, iterations = performRealFit(fitdata, freqData, params, acqNum, chirpNum,
                                                                  print_report)
    else:
        fitdata, params = setupRealFit(chirpData, freqData, guesses)
        pfit, perr, failed, fail_msg, iterations = performRealFit(fitdata, freqData, params, acqNum, chirpNum,
                                                                  print_report)

    return pfit, perr, failed, fail_msg, iterations


# Set up a fit where we fit the real and imaginary data
def setupRealFit(chirpData, freqData, xGuess):
    # Initalize parameters
    params = Parameters()

    dataAmp = np.abs(chirpData)

    # Get max value of data
    maxAmpVal = np.max(dataAmp)

    dataReal = np.real(chirpData)
    dataImag = np.imag(chirpData)

    fitdata = np.concatenate((dataReal, dataImag))

    params.add('res', value=xGuess[2], min=np.min(freqData), max=np.max(freqData))
    params.add('a', value=xGuess[0], min=0, max=maxAmpVal)
    params.add('phi', value=xGuess[1], min=0, max=2 * np.pi)
    params.add('Q', value=xGuess[3])
    # params.add('Offset', value=-np.mean(dataReal))

    return fitdata, params


# Perform the real and imaginary data fit
def performRealFit(data, freqData, params, acqNum, chirpNum, print_report):
    try:
        min_obj = lm.Minimizer(fitFunc, params, fcn_args=(freqData,), fcn_kws={
            'data': data})  # , fcn_args=(freqData,), fcn_kws={'data': data}, maxfev : 100000, ftol : 1e-9)
        # result = min_obj.leastsq(maxfev=100000, ftol=1e-7)
        # result = min_obj.minimize()
        # result = min_obj.minimize(fitFunc, params, method='leastsq', args=(freqData,), kws={'data': data}, maxfev=100000, ftol=1e-9)
        result = min_obj.minimize(maxfev=100000, ftol=1e-9)

        pfit = [result.params['a'].value, result.params['phi'].value, result.params['res'].value,
                result.params['Q'].value]  # ,result.params['Offset'].value]
        perr = [result.params['a'].stderr, result.params['phi'].stderr, result.params['res'].stderr,
                result.params['Q'].stderr]  # ,result.params['Offset'].stderr]

        iterations = result.nfev
        failed = False
        fail_msg = None
        if print_report:
            sys.stderr.write(lm.fit_report(result).encode('ascii'))
        # save_report(savepath, result, acqNum, chirpNum)
    except Exception as e:
        iterations = np.nan

        failed = True
        fail_msg = e

        perr = [np.inf, np.inf, np.inf, np.inf, np.inf]
        pfit = [np.inf, np.inf, np.inf, np.inf, np.inf]

        # save_report(savepath, '0', acqNum, chirpNum, msg=str(e))

    return pfit, perr, failed, fail_msg, iterations


# Set up a fit where we fit the amplitude data
def setupAmpFit(chirpData, freqData, xGuess):
    # Initalize parameters
    params = Parameters()

    dataAmp = np.abs(chirpData)
    dataPhase = np.angle(chirpData)

    # Get max value of data
    maxAmpVal = np.max(dataAmp)

    fitdata = dataAmp

    params.add('res', value=xGuess[2], min=np.min(freqData), max=np.max(freqData))
    params.add('a', value=xGuess[0], min=0, max=maxAmpVal)
    params.add('Q', value=xGuess[3])
    # params.add('Offset', value=-np.mean(dataAmp))

    return fitdata, params, dataPhase


# Perform the fit to the amplitude fit
def performAmpFit(data, freqData, params, dataPhase, acqNum, chirpNum, print_report=False):
    try:
        min_obj = lm.Minimizer(fitFuncAmp, params, fcn_args=(freqData,), fcn_kws={
            'data': data})  # , fcn_args=(freqData,), fcn_kws={'data': data}, maxfev : 100000, ftol : 1e-9)
        # result = min_obj.leastsq(maxfev=100000, ftol=1e-7)
        # result = min_obj.minimize()
        # result = min_obj.minimize(fitFunc, params, method='leastsq', args=(freqData,), kws={'data': data}, maxfev=100000, ftol=1e-9)
        result = min_obj.minimize(maxfev=100000, ftol=1e-9)

        res = result.params['res'].value
        res_loc = np.where(np.logical_and(freqData > result.params['res'].value - result.params['res'].stderr,
                                          freqData < result.params['res'].value + result.params['res'].stderr))
        angle_at_res = dataPhase[res_loc]
        phi = np.mean(angle_at_res)

        pfit = [result.params['a'].value, phi, result.params['res'].value,
                result.params['Q'].value]  # ,result.params['Offset'].value]
        perr = [result.params['a'].stderr, 0, result.params['res'].stderr,
                result.params['Q'].stderr]  # ,result.params['Offset'].stderr]
        if print_report:
            sys.stderr.write(lm.fit_report(result).encode('ascii'))
        iterations = result.nfev
        failed = False
        fail_msg = None
    except Exception as e:
        iterations = np.nan

        failed = True
        fail_msg = e

        perr = [np.inf, np.inf, np.inf, np.inf, np.inf]
        pfit = [np.inf, np.inf, np.inf, np.inf, np.inf]
    return pfit, perr, failed, fail_msg, iterations


# Save a fit report for the provided fit result
def save_report(savepath, result, acq, chirp, savename='fit_report.bin', msg=None):
    with open(savepath + savename, "ab") as myfile:
        myfile.write(("Acquisition, Chirp: " + str(acq) + ', ' + str(chirp) + '\n').encode('ascii'))
        if msg is None:
            myfile.write(lm.fit_report(result).encode('ascii'))
        else:
            myfile.write(msg.encode('ascii'))
        myfile.write('\n'.encode('ascii'))
        myfile.close()

    # f = open(savepath+savename, "w")
    # f.write("Acquisition, Chirp: "+str(acq)+', '+str(chirp)+'\n')
    # f.write(lm.fit_report(result))
    # f.close()


def limFitData(data, freq, low, high):
    upperLim = high
    lowerLim = low

    # We cant extend past the data we actuall have!!
    if upperLim > np.max(freq):
        upperLim = np.max(freq)
    if lowerLim < np.min(freq):
        lowerLim = np.min(freq)

    # Get the indicies where the data is within this range
    indicies = np.where(np.logical_and(freq > lowerLim, freq < upperLim))

    return data[indicies], freq[indicies]


# Fit a single chirp
#   acqData: data for an entire acquistion
#   lowFreq: the lower frequency limit
#   highFreq: the high frequency limit
#   remove_background: remove the linear background (True) or not (False) before fitting
#   smoothData: smooth the data (True) or not (False) before fitting
#   numAcq: the acquisition number
#   chirp: the chirp number to fit
#   fittype: Fit the real and imaginary data ('RealImag') or the amplitude data ('Amp')
def fitChirp(acqData, lowFreq, highFreq, chirpMap=None, remove_background=False, smoothData=False, numAcq=0, chirp=0,
             fittype='RealImag', print_report=False):
    # Extract chirp and freq axis
    chirpData = acqData.xs(chirp)

    flag = 0
    msg = 0

    phaseAdjust = False

    freq = chirpData.index.values

    # Scale data to uV
    chirpData = (10 ** 6) * chirpData.values

    # You can specify if you want to smooth the data prior to fitting
    if smoothData:
        chirpData, freq = smoothChirp(chirpData, freq)

    # Set up the data entry for this chirp/acq
    # See the ppt for the analysis overview for more
    # info on each entry
    if chirpMap is None:
        info = {}
        info['Packet'] = False
        info['PlotGroup'] = False
        info['Harmonic'] = False
        info['InOut'] = False
        info['DC'] = False
        info['Multiplier'] = False
    else:
        info = chirpMap.loc[chirp - 1, :]
        freq = np.multiply(freq, info['Harmonic'])

    # YOu can specify if you want to remove a linear curve from the data
    if remove_background:
        chirpData = remove_linear(chirpData)

    # Get a guess for the SHO fit
    xGuess = getSHOguess(chirpData, freq)
    xGuess = np.array(xGuess, dtype='float64')

    # Try to trim the data then fit, if it is a particularly bad response this will fail
    # In that case, just fit the entire data
    trimData, freqData, failed = attempt_trim(chirpData, freq, lowFreq, highFreq)

    # If the trimming failed, note it
    if failed:
        flag = 2
        msg = 'Trimming Failed'

    # Perform the fit
    pfit, perr, failed_fit, fail_msg, iterations = performFit(trimData, freqData, xGuess, fittype, numAcq, chirp,
                                                              print_report)

    if failed_fit:
        flag = 1
        msg = fail_msg

    data = [info['Packet'], info['PlotGroup'], info['Harmonic'], info['InOut'], info['DC'], info['Multiplier'], pfit[0],
            pfit[1], pfit[2], pfit[3], perr[0], perr[1], perr[2], perr[3], iterations, flag, msg]

    return data


# Fits an entire acquisition
#   responseFFT: the frequnecy response of the entire acquisition
#   numAcq: the acquisition number
#   grid: the grid size (if there is one)
#   lowFreq: the lower frequency limit
#   highFreq: the high frequency limit
#   smoothData: smooth the data (True) or not (False) before fitting
#   remove_background: remove the linear background (True) or not (False) before fitting
#   fittype: Fit the real and imaginary data ('RealImag') or the amplitude data ('Amp')

def shoFit(responseFFT, numAcq, chirpMap=None, grid=None, lowFreq=None, highFreq=None, smoothData=False,
           remove_background=False, fittype='RealImag'):
    # Get the names of the chirps
    chirps = responseFFT.index.get_level_values(0)
    chirpNames = chirps.unique()
    numChirps = len(chirpNames)

    # Extract Acquisition
    acqData = responseFFT

    # Initalize output data
    outData = []

    # For each chirp in acquisition
    for chirp in chirpNames:

        data = fitChirp(acqData, lowFreq, highFreq, chirpMap=chirpMap, remove_background=remove_background,
                        smoothData=smoothData,
                        numAcq=numAcq, chirp=chirp, fittype=fittype, print_report=False)

        try:
            outData = np.vstack([outData, data])
        except:
            outData = data

    # Then get the column labels
    indexChirps = chirpNames.values

    if grid is None:
        acqArray = np.repeat(numAcq, len(indexChirps))
    elif grid != 0:
        line = numAcq // grid

        if np.mod(line, 2):
            newAcq = (1 + 2 * line) * grid - numAcq - 1
        else:
            newAcq = numAcq

        acqArray = np.repeat(newAcq, len(indexChirps))
    else:
        acqArray = np.repeat(numAcq, len(indexChirps))

    # Get the multi index from the products of these two
    index = pd.MultiIndex.from_arrays([acqArray, indexChirps], names=['Acq', 'ChirpNum'])

    # Get the multi columns from the products of these two
    columns = pd.Index(
        ['Packet', 'PlotGroup', 'Harmonic', 'InOut', 'DC', 'Multiplier', 'Amp', 'Phase', 'Res', 'Q', 'errA', 'errP',
         'errRes', 'errQ', 'Iter', 'Flag', 'Msg'], names=['Values'])

    # Create the total output data
    extractedData = pd.DataFrame(outData, index=index, columns=columns)

    return extractedData


# Wrapper around shoFit. Pull the data from the HDF store, get the spectrum, and pass it to shoFit
def shoFitAcq(acqNum, fftFileName, chirpMap=None, grid=None, lowFreq=None, highFreq=None, smooth=None,
              remove_background=False, fittype='RealImag'):
    # Get a path to the fft information
    store = pd.HDFStore(fftFileName)

    # extract the data for this acquisition
    spec = store['Acq' + str(acqNum)]

    # Call the SHO fitter for each chirp
    extracted = shoFit(spec, acqNum, chirpMap=chirpMap, grid=grid, lowFreq=lowFreq, highFreq=highFreq,
                       smoothData=smooth,
                       remove_background=remove_background, fittype=fittype)

    return extracted


# Run shoFitAcq for an entire data set with numAcq number of acquisitions.
def fitdataset(fftFileName, waveSpecFileName, numAcq, grid=None, low=None, high=None, smooth=False,
               remove_background=False, fittype='RealImag'):
    chirpMap = generateChirpMap(waveSpecFileName)

    shoPartial = partial(shoFitAcq, fftFileName=fftFileName, chirpMap=chirpMap, grid=grid, lowFreq=low, highFreq=high,
                         smooth=smooth, remove_background=remove_background, fittype=fittype)

    results = shoPartial(0)

    for i in np.arange(1, numAcq):
        res = shoPartial(i)
        results = pd.concat([results, res], axis=0)
        sys.stderr.write('\rdone {0:%}'.format(i / numAcq))

    return results


def generateChirpMap(waveformSpecPath):
    # Read in the waveform specification file
    waveSpec = pd.read_csv(waveformSpecPath)

    # Get the column labels
    cols = waveSpec.columns.values

    # Get the number of chirps and packets
    numChirps = int(np.sum(waveSpec.loc[:, cols[0]].values))
    numPackets = waveSpec.shape[0]

    # initialize the chirp map
    chirpMap = np.empty([numChirps, 6])

    # chirp couter
    chirpCount = 0
    prevDC = 0

    for i in np.arange(0, numPackets):

        # Get the info on the packet and the
        packInfo = waveSpec.loc[i, :]
        packetSize = packInfo[cols[0]]
        dcField = packInfo[cols[1]]
        multiplier = packInfo[cols[2]]
        harmonic = packInfo[cols[3]]
        plotGroup = packInfo[cols[4]]

        # For each chirp in the packet
        for j in np.arange(0, packetSize):

            currentRow = int(j + chirpCount)

            chirpMap[currentRow, 0] = i
            chirpMap[currentRow, 1] = plotGroup
            chirpMap[currentRow, 2] = harmonic
            chirpMap[currentRow, 5] = multiplier

            # If out of field
            if dcField == 0:
                chirpMap[currentRow, 3] = 0
                chirpMap[currentRow, 4] = prevDC
            else:
                chirpMap[currentRow, 3] = 1
                chirpMap[currentRow, 4] = dcField
                prevDC = dcField

        # Keep track of the number of chirps
        chirpCount = chirpCount + packetSize

    # Then get the row labels
    chirps = np.repeat('Chirp', numChirps)
    numbers = np.arange(1, numChirps + 1)
    numbers = list(map(str, numbers))
    comb = list(zip(chirps, numbers))
    joined_data = (''.join(w) for w in comb)
    rows = list(joined_data)

    finalMap = pd.DataFrame(chirpMap, index=np.arange(0, numChirps),
                            columns=['Packet', 'PlotGroup', 'Harmonic', 'InOut', 'DC', 'Multiplier'])

    return finalMap


def redo_fitting_chirp(fftFileName, waveSpecFileName, numAcq, chirpnum, report=False, plot=False,
                       low=None, high=None, smooth=False, remove_background=False, fittype='RealImag'):
    # Get a path to the fft information
    store = pd.HDFStore(fftFileName)

    chirpMap = generateChirpMap(waveSpecFileName)

    # extract the data for this acquisition
    spec = store['Acq' + str(numAcq)]

    results = fitChirp(spec, low, high, chirpMap=chirpMap, remove_background=remove_background, smoothData=smooth,
                       numAcq=numAcq, chirp=chirpnum, fittype=fittype, print_report=report)

    if plot:
        plot_chirp_redo(results, chirpnum, spec,fittype=fittype)

    return results

def plot_chirp_redo(results, chirpnum, spec, fittype='RealImag'):

    A, Ph, Res, Q = results[6:10]

    chirpData=signal.detrend(spec.xs(chirpnum))
    amp = np.abs(chirpData)
    phase = np.angle(chirpData)
    freq=spec.xs(chirpnum).index.values

    if fittype is 'RealImag':
        fitted = complexGaus(freq,A,Ph,Res,Q)
        fitted_amp = np.abs(fitted)
        fitted_angle = np.angle(fitted)

        fig, ax1 = plt.subplots(figsize = (8,8))

        color = 'tab:red'
        ax1.set_xlabel('Freq (Hz)')
        ax1.set_ylabel('Amp (V)',color=color)
        ax1.plot(freq, amp, '-',color=color)
        ax1.plot(freq,fitted_amp/1e6, '--k')
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        color = 'tab:blue'
        ax2.set_ylabel('Phase', color=color)  # we already handled the x-label with ax1
        ax2.plot(freq, phase, '-', color=color)
        ax2.plot(freq,fitted_angle, '--k')

        fig.tight_layout(pad=3.0)  # otherwise the right y-label is slightly clipped
        plt.show()

        fig, ax1 = plt.subplots(figsize = (8,8))

        color = 'tab:red'
        ax1.set_xlabel('Freq (Hz)')
        ax1.set_ylabel('Acos(Phi) AKA Real',color=color)
        #ax1.plot(freq, signal.detrend(amp*np.cos(phase)), '.',color=color)
        ax1.plot(freq, chirpData, '.',color=color)
        ax1.plot(freq,(np.abs(fitted)/1e6)*np.cos(fitted_angle),'--',color='orange')
        ax1.tick_params(axis='y', labelcolor=color)

    else:
        fitted_amp = complexGausAmp(freq,A,Res,Q)

        fig, ax1 = plt.subplots(figsize = (8,8))

        color = 'tab:red'
        ax1.set_xlabel('Freq (Hz)')
        ax1.set_ylabel('Amp (V)',color=color)
        ax1.plot(freq, amp, '-',color=color)
        ax1.plot(freq,fitted_amp/1e6, '--k')
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        color = 'tab:blue'
        ax2.set_ylabel('Phase', color=color)  # we already handled the x-label with ax1
        ax2.plot(freq, phase, '-', color=color)
        ax2.plot(freq,np.repeat(Ph,len(freq)), '--k')

        fig.tight_layout(pad=3.0)  # otherwise the right y-label is slightly clipped
        plt.show()