#####Standard packages we import for use in most code
import numpy as np  # for numerical operations
import pandas as pd  # for data reading/structure
import scipy.signal as signal
import scipy.io
from scipy.signal import argrelextrema
from functools import partial
import multiprocessing
from multiprocessing import Pool
from lmfit import Model, Parameters, minimize
import lmfit as lm
from tqdm.notebook import tqdm
from tqdm.contrib import tenumerate
from statsmodels.stats.weightstats import DescrStatsW
from scipy import stats


##### SHO Fitting Functions

#####Use waveform to gnerate fitting information for eqch chirp
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

    # chirp counter
    chirpCount = 0
    prevDC = 0

    for i in np.arange(0, numPackets):

        # Get the info on the packet
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
    chirps = np.repeat("Chirp", numChirps)
    numbers = np.arange(1, numChirps + 1)
    numbers = list(map(str, numbers))
    comb = list(zip(chirps, numbers))
    joined_data = ("".join(w) for w in comb)
    rows = list(joined_data)

    finalMap = pd.DataFrame(
        chirpMap,
        index=np.arange(1, numChirps + 1),
        columns=["Packet", "PlotGroup", "Harmonic", "InOut", "DC", "Multiplier"],
    )

    return finalMap


# Pass the frequency spectrum for a chirp and the frequnecy axis, and this function will return a guess for the
# SHO fit
def getSHOguess(chirpData, freq, res_prev=np.nan):
    # Get amplitude and phase
    amp = np.abs(chirpData)
    phase = np.angle(chirpData)

    # -----------------Get some guesses for the fitting-----------------
    # if res_prev==np.nan:
    # resGuess = np.argmax(amp)
    # else:
    # resGuess =min(freq,key=lambda x:abs(res_prev))
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

    if Q > 2000:
        Q = 2000

    a = ampGuess / Q

    xGuess = [a, phi, res, Q]

    return xGuess


# Complex gaussian function describing a damped-driven oscillator
def complexGaus(x, a, phi, res, Q):
    func = a * np.exp(1j * phi) * res**2 / (x**2 - 1j * x * res / Q - res**2)
    return func


# Amplitude of the complex gaussian
def complexGausAmp(x, a, res, Q):
    func = (a * res**2) / (np.sqrt(((x**2 - res**2) ** 2) + ((x**2 * res**2) / (Q**2))))
    return func


# A standard moving average filter for data smoothing
def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1 :] / n


# Function that is used for fitting. It calucaltes the complex gaussian and returns the error for given data
def fitFunc(pars, x, data=None):
    vals = pars.valuesdict()
    a = vals["a"]
    phi = vals["phi"]
    res = vals["res"]
    Q = vals["Q"]
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
    a = vals["a"]
    # phi=vals['phi']
    res = vals["res"]
    Q = vals["Q"]
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
    chirpData = signal.detrend(chirpData)
    return chirpData


## This function finds local noisepionts in the raw amplitude response in a region outside of of the user defined SHO fit limits


def find_region_noise(chirpData, freq, limit, q=(0.5, 0.95)):
    ####Extract the noisey part sprectrum i.e. data outside the fitting limits

    ###Determine if limit is high or low
    test = int(len(freq) / 2)  ######index of middle of frequncy range
    if limit < freq[test]:
        # Get the indicies where the data is below limit
        indicies = np.where(freq < limit)
    else:
        # Get the indicies where the data is above the limit
        indicies = np.where(freq > limit)

    ###Store data values the noise range
    noise = np.zeros(len(chirpData))
    noise[:] = np.nan  #####Start with nan values

    ###Same shape as data (plotting); contains nan values
    noise[indicies] = np.abs(chirpData)[
        indicies
    ]  #######finds outliers from the amplitude of response

    ###For calculations; no nans
    noise_vals = noise[np.isfinite(noise)]

    ######Find the distrobution of the noise
    n, bins = np.histogram(noise_vals, bins="sqrt")
    bins = 0.5 * (bins[1:] + bins[:-1])
    wq = DescrStatsW(data=bins, weights=n)  # compute wighted statistic

    ####Mask Noise values ouside user defined quartiles;
    # use this mask for further calculations (Background Removal or Signal detection)
    high_off = wq.quantile(probs=q[1], return_pandas=False)[0]
    low_off = wq.quantile(probs=q[0], return_pandas=False)[0]
    noise_mask = np.logical_and(noise < high_off, noise > low_off)
    return noise_mask  ######return noise values,the cut_off amplitude value and mask for plotting


###This function removes a global linear background subtraction using the noise masks returned from the find_region_noise() function
def calc_lin_background(chirpData, freq, high_mask, low_mask):
    freq_vals = np.concatenate((freq[low_mask], freq[high_mask]))
    noise_vals = np.concatenate((chirpData[low_mask], chirpData[high_mask]))
    lin_result = stats.linregress(freq_vals, noise_vals)
    fit = lin_result.slope * (freq) + lin_result.intercept
    bgs = chirpData - fit
    return bgs


# This funcion "detrends" the data prior to fitting using asymmetrically reweighted penalized least squares smoothing ref DOI 10.1039/c4an01061b
def remove_arPLS(chirpData, freq):
    freq = freq
    roi = make_roi(freq, high, low)
    ### Remove background from real portion of data
    real = np.real(chirpData)
    ycalc_arpls_real, base_arpsl_real = rampy.baseline(
        freq, real, roi, "arPLS", lam=10**6, ratio=0.005
    )
    ### Remove background from imaginary portion of data
    im = np.imag(chirpData)
    ycalc_arpls_im, base_arpsl_im = rampy.baseline(freq, im, roi, "arPLS", lam=10**6, ratio=0.005)
    ### Recombine data
    arpls = ycalc_arpls_real.flatten() + 1j * ycalc_arpls_im.flatten()
    return arpls


# This function tries to trim the data according to the user-provided low and high frequency limits
def attempt_trim(chirpData, freq, lowFreq, highFreq):
    # Try to trim the data according to the provided freqeuncy limits.
    # If it fails recore the exception
    try:
        # Trim unecessary data
        if lowFreq is not None:
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
    if fitType == "Amp":
        fitdata, params, phase = setupAmpFit(chirpData, freqData, guesses)
        pfit, perr, failed, fail_msg, iterations = performAmpFit(
            fitdata, freqData, params, phase, acqNum, chirpNum, print_report
        )
    elif fitType == "RealImag":
        fitdata, params = setupRealFit(chirpData, freqData, guesses)
        pfit, perr, failed, fail_msg, iterations = performRealFit(
            fitdata, freqData, params, acqNum, chirpNum, print_report
        )
    else:
        fitdata, params = setupRealFit(chirpData, freqData, guesses)
        pfit, perr, failed, fail_msg, iterations = performRealFit(
            fitdata, freqData, params, acqNum, chirpNum, print_report
        )

    return pfit, perr, failed, fail_msg, iterations


# Set up a fit where we fit the real and imaginary data
def setupRealFit(chirpData, freqData, xGuess, AmpVal=None):
    # Initalize parameters
    params = Parameters()

    dataAmp = np.abs(chirpData)

    # Get max value of data
    maxAmpVal = np.max(dataAmp)

    dataReal = np.real(chirpData)
    dataImag = np.imag(chirpData)

    fitdata = np.concatenate((dataReal, dataImag))

    ###Amp Val is given for low SNR datapoints in the first quarter cycle to ensure a value for information recovery
    if AmpVal is not None:
        params.add("a", value=AmpVal, vary=False)
    else:
        try:
            params.add("a", value=xGuess[0], min=0, max=maxAmpVal)
        except Exception as e:
            params.add("a", value=xGuess[0], min=0, max=0.05)

    params.add("res", value=xGuess[2], min=np.min(freqData), max=np.max(freqData))
    params.add("phi", value=xGuess[1], min=0, max=2 * np.pi)
    params.add("Q", value=xGuess[3], min=0, max=10000)
    return fitdata, params


# Perform the real and imaginary data fit
def performRealFit(data, freqData, params, acqNum, chirpNum, print_report):
    try:
        min_obj = lm.Minimizer(fitFunc, params, fcn_args=(freqData,), fcn_kws={"data": data})
        result = min_obj.minimize(max_nfev=100000, ftol=1e-9)

        pfit = [
            result.params["a"].value,
            result.params["phi"].value,
            result.params["res"].value,
            result.params["Q"].value,
        ]  # ,result.params['Offset'].value]
        perr = [
            result.params["a"].stderr,
            result.params["phi"].stderr,
            result.params["res"].stderr,
            result.params["Q"].stderr,
        ]  # ,result.params['Offset'].stderr]

        iterations = result.nfev
        residual = result.residual
        failed = False
        fail_msg = None
        if print_report:
            print(lm.fit_report(result))
        # save_report(savepath, result, acqNum, chirpNum)
    except Exception as e:
        iterations = np.nan

        failed = True
        fail_msg = e

        perr = [np.nan, np.nan, np.nan, np.nan]
        pfit = [np.nan, np.nan, np.nan, np.nan]

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

    params.add("res", value=xGuess[2], min=np.min(freqData), max=np.max(freqData))
    params.add("a", value=xGuess[0], min=0, max=maxAmpVal)
    params.add("Q", value=xGuess[3])
    # params.add('Offset', value=-np.mean(dataAmp))

    return fitdata, params, dataPhase


# Perform the fit to the amplitude fit
def performAmpFit(data, freqData, params, dataPhase, acqNum, chirpNum, print_report=False):
    try:
        min_obj = lm.Minimizer(fitFuncAmp, params, fcn_args=(freqData,), fcn_kws={"data": data})
        result = min_obj.minimize(max_nfev=100000, ftol=1e-9)

        res = result.params["res"].value
        res_loc = np.where(
            np.logical_and(
                freqData > result.params["res"].value - result.params["res"].stderr,
                freqData < result.params["res"].value + result.params["res"].stderr,
            )
        )
        angle_at_res = dataPhase[res_loc]
        phi = np.mean(angle_at_res)

        pfit = [
            result.params["a"].value,
            phi,
            result.params["res"].value,
            result.params["Q"].value,
        ]  # ,result.params['Offset'].value]
        perr = [
            result.params["a"].stderr,
            np.nan,
            result.params["res"].stderr,
            result.params["Q"].stderr,
        ]  # ,result.params['Offset'].stderr]
        if print_report:
            print(lm.fit_report(result))
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
def save_report(savepath, result, acq, chirp, savename="fit_report.bin", msg=None):
    with open(savepath + savename, "ab") as myfile:
        myfile.write(("Acquisition, Chirp: " + str(acq) + ", " + str(chirp) + "\n").encode("ascii"))
        if msg is None:
            myfile.write(lm.fit_report(result).encode("ascii"))
        else:
            myfile.write(msg.encode("ascii"))
        myfile.write("\n".encode("ascii"))
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
    indicies = np.where(np.logical_and(freq >= lowerLim, freq <= upperLim))

    return data[indicies], freq[indicies]


# Fit a single chirp
# acqData: data for an entire acquistion
# lowFreq: the lower frequency limit
# highFreq: the high frequency limit
# acqNum: the acquisition number
# chirp: the chirp number to fit
# fittype: Fit the real and imaginary data ('RealImag') or the amplitude data ('Amp')
# tempguess is a two element list containing the 1st and 2nd nearest nieghbor SHO values
def fitChirp(
    acqData,
    lowFreq,
    highFreq,
    chirpMap=None,
    remove_background=True,
    acqNum=0,
    chirp=0,
    fittype="RealImag",
    skip_pg=9.0,
    temp_guess=None,
):

    ## Find experimental information about the applied signal
    if chirpMap is None:
        info = {}
        info["Packet"] = False
        info["PlotGroup"] = False
        info["Harmonic"] = False
        info["InOut"] = False
        info["DC"] = False
        info["Multiplier"] = False
    else:
        ## Find experimental information about the applied signal
        info = chirpMap.loc[chirp, :]
        # print(info)

    ####Skip dead Chirps
    if info["PlotGroup"] == 9.0:
        data = [
            info["Packet"],
            info["PlotGroup"],
            info["Harmonic"],
            info["InOut"],
            info["DC"],
            info["Multiplier"],
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        ]
    else:
        # Extract raw data anf frequency values
        Data = acqData.xs(chirp).values
        freq = acqData.xs(chirp).index.values

        ##Trim
        trim, trimfreq, failed = attempt_trim(Data, freq, lowFreq, highFreq)

        # Remove a linear background from the Local data
        if remove_background:
            bgs = remove_linear(trim)
        else:
            bgs = trim

        # Get a guess for the SHO fit unless one is provided
        if temp_guess is None:
            xGuess = getSHOguess(bgs, trimfreq)
        else:
            xGuess = temp_guess[0]
            if np.isnan(xGuess).all():
                xGuess = temp_guess[1]
                if np.isnan(xGuess).all():
                    xGuess = getSHOguess(bgs, trimfreq)
        xGuess = np.array(xGuess, dtype="float64")

        ####### Set up a fit where we fit the real and imaginary data
        fitData, params = setupRealFit(bgs, trimfreq, xGuess, AmpVal=None)
        # Perform the real and imaginary data fit
        pfit, perr, failed, fail_msg, iterations = performRealFit(
            fitData, trimfreq, params, acqNum, chirp, False
        )
        data = [
            info["Packet"],
            info["PlotGroup"],
            info["Harmonic"],
            info["InOut"],
            info["DC"],
            info["Multiplier"],
            pfit[0],
            pfit[1],
            pfit[2],
            pfit[3],
            perr[0],
            perr[1],
            perr[2],
            perr[3],
        ]

    return data


def shoFit(
    responseFFT,
    acqNum,
    chirpMap=None,
    grid=None,
    lowFreq=None,
    highFreq=None,
    fittype="RealImag",
    break_chirp=None,
):
    # Extract Acquisition
    acqData = responseFFT

    # Get the names of the chirps
    chirps = responseFFT.index.get_level_values(0)
    chirpNames = chirps.unique()
    numChirps = len(chirpNames)

    # Then get the column labels
    indexChirps = chirpNames.values
    acqArray = np.repeat(acqNum, len(indexChirps))

    if grid is None:
        acqArray = np.repeat(acqNum, len(indexChirps))
    elif grid != 0:
        line = acqNum // grid

        if np.mod(line, 2):
            newAcq = (1 + 2 * line) * grid - acqNum - 1
        else:
            newAcq = acqNum

        acqArray = np.repeat(newAcq, len(indexChirps))
    else:
        acqArray = np.repeat(acqNum, len(indexChirps))

    # Get the multi index from the products of these two
    index = pd.MultiIndex.from_arrays([acqArray, indexChirps], names=["Acq", "ChirpNum"])
    # Get the multi columns from the products of these two
    columns = pd.Index(
        [
            "Packet",
            "PlotGroup",
            "Harmonic",
            "InOut",
            "DC",
            "Multiplier",
            "Amp",
            "Phase",
            "Res",
            "Q",
            "errA",
            "errP",
            "errRes",
            "errQ",
        ]
    )

    # Initalize output data
    outData = pd.DataFrame(np.full((len(index), len(columns)), -1), index=index, columns=columns, dtype="float64")

    if break_chirp is None:
        # For each chirp in acquisition
        for chirp in chirpNames:
            data = fitChirp(
                acqData,
                lowFreq,
                highFreq,
                chirpMap=chirpMap,
                acqNum=acqNum,
                chirp=chirp,
                fittype=fittype,
            )
            outData.loc[(acqNum, chirp), :] = data
    else:
        chirp_loop1 = chirpNames[break_chirp:]
        #####first fit all chirps after the first quarter in the forward direction
        for i, chirp in enumerate(chirp_loop1):
            if i > 3:
                data = fitChirp(
                    acqData,
                    lowFreq,
                    highFreq,
                    chirpMap=chirpMap,
                    acqNum=acqNum,
                    chirp=chirp,
                    fittype=fittype,
                )
                outData.loc[(acqNum, chirp), :] = data
            else:
                data = fitChirp(
                    acqData,
                    lowFreq,
                    highFreq,
                    chirpMap=chirpMap,
                    acqNum=acqNum,
                    chirp=chirp,
                    fittype=fittype,
                )

            outData.loc[(acqNum, chirp), :] = data

        ##### Next fit all chrips before the break in the backwards direction using the nearest nieghbors for inital x_guesses
        chirp_loop2 = np.flip(chirpNames[:break_chirp])
        for chirp in chirp_loop2:
            data = fitChirp(
                acqData,
                lowFreq,
                highFreq,
                chirpMap=chirpMap,
                acqNum=acqNum,
                chirp=chirp,
                fittype=fittype,
                temp_guess=[
                    outData.loc[(acqNum, chirp + 2), ("Amp", "Phase", "Res", "Q")].values,
                    outData.loc[(acqNum, chirp + 4), ("Amp", "Phase", "Res", "Q")].values,
                ],
            )

            outData.loc[(acqNum, chirp), :] = data

    return outData


# Wrapper around shoFit. Pull the data from the HDF store, get the spectrum, and pass it to shoFit, and optinally export reconstruction files for matrix completion by BayeSMG (run seperately in MatLab)
def shoFitAcq(
    acqNum,
    fftFileName,
    chirpMap=None,
    grid=None,
    lowFreq=None,
    highFreq=None,
    fittype="RealImag",
    break_chirp=None,
    exportPath=None,
    keepFirst=None,
    errQ_Ratio=None,
):
    # Get a path to the fft information
    store = pd.HDFStore(fftFileName)

    # extract the data for this acquisition
    spec = store["Acq" + str(acqNum)]

    extracted = shoFit(
        spec,
        acqNum,
        chirpMap=chirpMap,
        grid=None,
        lowFreq=lowFreq,
        highFreq=highFreq,
        fittype=fittype,
        break_chirp=break_chirp,
    )

    if exportPath is not None:
        ######Grab the raw complex values data and create new datafame for reconstructions
        raw_df = chirp_df(fftFileName, AcqNum)
        raw_df = raw_df.loc[
            keep[0] :,
            np.logical_and(
                raw_df.columns.values <= highFreq,
                raw_df.columns.values >= lowFreq,
            ),
        ]
        export_df = pd.DataFrame(index=raw_df.index, columns=raw_df.columns, dtype="float64")
        ####Find outliers based on Q error or failed fits
        mask = find_outliers_Q(extracted, errQ_Ratio)
        for pnt, chirp in raw_df.index:
            A, Ph, Res, Q, errQ = extracted.loc[(pnt, chirp), ["Amp", "Phase", "Res", "Q", "errQ"]].values
            fit = complexGaus(raw_df.columns.values, A, Ph, Res, Q)
            export_df.loc[(pnt, chirp), :] = fit
            ##### keep initial chirps in the first quarter cycle if Low SNR
            if keepFirst is not None:
                if (errQ is None) & (chirp in keepFirst):
                    mask.loc[(pnt, chirp)] = False
                elif ((errQ / Q) > errQ_Ratio) & (chirp in keepFirst):
                    mask.loc[(pnt, chirp)] = False

        ####Remove unreliable chirps
        export_df[mask.values] = np.nan + 1j * np.nan
        # Export out puts in fit_pnt dictionary
        recon_dict = {
            "Real": pd.DataFrame(np.real(export_df), index=raw_df.index, columns=raw_df.columns),
            "Im": pd.DataFrame(np.imag(export_df), index=raw_df.index, columns=raw_df.columns),
            "SHO": extracted, index=extracted.index, columns=extracted.columns)
        }
        ####Export to CSV
        for (key, df_) in list(recon_dict.items()):
            save_temp = savePath + "Loc_" + str(pnt) + "_" + key + ".csv"
            df_.to_csv(save_temp, index=[0, 1])

    return extracted


# Run shoFitAcq for an entire data set with numAcq number of acquisitions.
def fitdataset(
    fftFileName,
    waveSpecFileName,
    numAcq,
    grid=None,
    low=None,
    high=None,
    fittype="RealImag",
    break_chirp=None,
    exportPath=None,
    keepFirst=None,
    errQ_Ratio=None,
):
    chirpMap = generateChirpMap(waveSpecFileName)

    shoPartial = partial(
        shoFitAcq,
        fftFileName=fftFileName,
        chirpMap=chirpMap,
        grid=grid,
        lowFreq=low,
        highFreq=high,
        fittype=fittype,
        break_chirp=break_chirp,
        exportPath=exportPath,
        keepFirst=None,
    )
    
    ####Set up the index
    numChirps = chirpMap.shape[0]
    acq_array = np.repeat(np.arange(0, numAcq), numChirps)
    chirp_array = np.tile(np.arange(0, numChirps), numAcq)
    index = pd.MultiIndex.from_arrays([acq_array, chirp_array], names=["Acq", "ChirpNum"])
    ####Set up the columns
    columns = pd.Index(
        [
            "Packet",
            "PlotGroup",
            "Harmonic",
            "InOut",
            "DC",
            "Multiplier",
            "Amp",
            "Phase",
            "Res",
            "Q",
            "errA",
            "errP",
            "errRes",
            "errQ",
        ]
    )

    #####Make Results Dataframe 
    results = pd.DataFrame(np.full((len(index), len(columns)), -1), index=index, columns=columns)

    for i in np.arange(0, numAcq):
        res = shoPartial(i)
        results.loc[i] = res
    return results


def chirp_df(fftFileName, pnt):
    # Get a path to the fft information
    store = pd.HDFStore(fftFileName)
    # extract the data for this acquisition
    data = store["Acq" + str(pnt)].copy()
    temp_ind = data.index
    acqArray = np.repeat(pnt, len(temp_ind))
    data.index = pd.MultiIndex.from_arrays([acqArray, temp_ind], names=["Acq", "ChirpNum"])
    return data.iloc[:-1]


######Functions that find outliers due to poor SHO fits
def find_outliers_Q(df, err_ratio):
    #####Make Copy of Q values data frame
    y = df["Q"].copy()  ####Q values
    err_q = df["errQ"].copy()  ####fitting errors for Q

    ######Find err ratio greater than 40 %
    ratios = np.divide(err_q, y)
    err_mask = np.abs(ratios) > err_ratio
    ######nan in errQ indicates poor SHO fit
    err_q_null = np.isnan(err_q)
    #####Mask Q values equal to NAN or that have error values greater than 40% of Q value
    total_mask = np.logical_or(err_mask, err_q_null)  # True means poor SHO fit
    return total_mask


