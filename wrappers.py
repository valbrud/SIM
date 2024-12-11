"""
wrappers.py

This module contains wrapper functions for Fourier transforms to make shifts automatically and
make it possible to switch between their implementations.

Functions:
    wrapped_fftn: Wrapper for the FFTN function.
    wrapped_ifftn: Wrapper for the IFFTN function.
"""

import numpy as np

# Wrappers to avoid shifting the arrays every time DFT is used
def wrapper_ft(ft):
    """
    Wrapper for the Fourier transform functions to make shifts automatically.
    Currently based on numpy fft implementation.
    """
    def wrapper(arrays, *args, **kwargs):
        return np.fft.fftshift(ft(np.fft.ifftshift(arrays), *args, **kwargs))

    return wrapper


wrapped_fft = wrapper_ft(np.fft.fft)
wrapped_ifft = wrapper_ft(np.fft.ifft)

wrapped_fftn = wrapper_ft(np.fft.fftn)
wrapped_ifftn = wrapper_ft(np.fft.ifftn)
