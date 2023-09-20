import numpy as np

#Wrappers to avoid shifting the arrays every time DFT is used
def wrapper_ft(ft):
    def wrapper(arrays, *args, **kwargs):
        return np.fft.fftshift(ft(np.fft.fftshift(arrays), *args, **kwargs))
    return wrapper


wrapped_fft = wrapper_ft(np.fft.fft)
wrapped_ifft = wrapper_ft(np.fft.ifft)

wrapped_fftn = wrapper_ft(np.fft.fftn)
wrapped_ifftn = wrapper_ft(np.fft.ifftn)


