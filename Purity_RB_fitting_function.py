# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 14:53:24 2025

@author: kbharadwaj
"""
import numpy as np
from scipy.optimize import curve_fit

def exponential_decay_1D_purity(t, decay, amplitude, offset):
    """Model for an exponential decay."""
    return amplitude * np.exp(-(t-1) / (decay)) + offset

def fit_purity(xdata_pre,ydata):
    num_samples = np.compress(xdata_pre==xdata_pre[-1], xdata_pre).size
    num_qubits = 1
    dim = 2**num_qubits
    popt, pcov = curve_fit(exponential_decay_1D_purity, xdata_pre, ydata, p0=[0.5 * xdata_pre[-1], ydata[0], ydata[0]-ydata[-1]], bounds=([0,0,0],[xdata_pre[-1],ydata[0],1]))
    gamma = np.exp(-1.0 / popt[0])
    error = (dim - 1) / dim * (1 - gamma ** (1 / 2))
    fidelity = 1 - error
    fidelity_error = get_fidelity_error(xdata_pre, ydata, num_samples, popt)
    return [fidelity, fidelity_error]

def get_fidelity_error(xdata_pre, ydata, num_samples, popt, confidence_interval = 0.9):
    num_qubits = 1
    dim = 2**num_qubits
    A = popt[1]
    alpha = np.exp(-1.0 / popt[0])
    e0 = popt[2]
    xdata=xdata_pre[0::num_samples]
    N = len(xdata)
    D = 3
    # construct the Q matrix (Hessian)
    jacobian = np.array(
        [  # df/d_alpha, df/d_A, df/d_e0
            [n_seq/2 * A * alpha ** (n_seq - 1)/2, alpha ** n_seq/2, 1.0]
            for n_seq in xdata
        ]
    )
    R = np.dot(jacobian.T, jacobian)
    try:
        Q = sl.inv(R)
    except np.linalg.LinAlgError:
        print('Fidelity error calculation failed.')
        return 1.
    # construct the t-distribution confidence interval
    ci_t = st.t.interval(confidence_interval, N - D)
    ci_t_upper = ci_t[1]
    # construct the standard error
    model_data = (A * (alpha ** xdata/2)) + e0
    ydata = ydata.reshape(int(len(ydata)/num_samples),num_samples)
    diffs = np.array(np.average(ydata, axis=1) - model_data)
    std_err = np.sqrt(np.dot(diffs.T, diffs) / num_samples)
    s = std_err / np.sqrt(N - D)
    # put together to calculate error
    sqrtQ = np.sqrt(Q[0][0])
    alpha_error = ci_t_upper * s * sqrtQ / 2.
    fidelity_error = ((dim - 1) / (dim)) * alpha_error
    fidelity_error = 0
    return fidelity_error