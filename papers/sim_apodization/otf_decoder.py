import base64
import xml.etree.ElementTree as ET
from pathlib import Path
import re
import numpy as np
import matplotlib.pyplot as plt

def _decode_band(node: ET.Element, samples: int) -> np.ndarray:
    """
    Decode one <band-*> element into a complex-valued 1-D NumPy array
    (length = *samples*).
    """
    # remove the BASE64: … :END64 wrappers and whitespace
    txt = re.sub(r"\s+", "", node.text.strip())
    if txt.startswith("BASE64:"):
        txt = txt[7:]
    if txt.endswith(":END64"):
        txt = txt[:-6]

    raw = base64.b64decode(txt)

    # --- this is the crucial fix – read the floats as **big endian** ---
    floats = np.frombuffer(raw, dtype=">f4").astype(np.float32)

    if floats.size != 2 * samples:
        raise ValueError(
            f"Expected {2*samples} floats, got {floats.size} in {node.tag}"
        )

    return floats[::2] + 1j * floats[1::2]


# ---------- public function --------------------------------------------------
def load_fairsim_otf(path_or_str, out_size: int = 512, fftshift: bool = True):
    """
    Load a fairSIM <otf2d> file and return it as a centred Cartesian array.

    Parameters
    ----------
    path_or_str : str | Path
        File name of the XML/OTF or the XML string itself.
    out_size : int, default 512
        Size (N) of the returned N×N grid.
    fftshift : bool, default True
        If True, low spatial frequencies are placed in the centre
        (same view as in the Java plugin after `Transforms.swapQuadrant`).

    Returns
    -------
    cube : (bands, N, N) complex64 ndarray
    """
    # ------------------------------------------------------------------ parse
    if "<fairsim" in str(path_or_str):
        root = ET.fromstring(path_or_str)
    else:
        root = ET.parse(Path(path_or_str)).getroot()

    data = root.find(".//otf2d/data")

    cycles = float(data.findtext("cycles"))       # Δk (cycles/µm) between samples
    samples = int(data.findtext("samples"))
    n_bands = int(data.findtext("bands"))

    # ----------------------------------------------------------- decode bands
    radial = np.stack(
        [_decode_band(data.find(f"band-{b}"), samples) for b in range(n_bands)]
    )  # → (bands, samples)

    # -------------------------------------------------------- cartesian grid
    pix_size = cycles / 2.0                       # see Java: setPixelSize(c/2)

    N = out_size
    half = N // 2
    y, x = np.mgrid[-half:half, -half:half]       # centred pixel coordinates
    r = np.hypot(x, y) * pix_size                # radial frequency (cycles µm⁻¹)

    r_support = cycles * np.arange(samples)

    idx = np.clip((r / cycles).astype(np.int32), 0, samples - 2)
    frac = (r - idx * cycles) / cycles

    cube = np.empty((n_bands, N, N), np.complex64)
    for b in range(n_bands):
        band = radial[b]
        vals = band[idx] * (1 - frac) + band[idx + 1] * frac
        vals[r >= r_support[-1]] = 0.0            # outside the measured range
        cube[b] = vals

    if fftshift:
        cube = np.fft.fftshift(cube, axes=(-2, -1))

    return cube


# -------------------------------------------------------------------------
# Example usage ------------------------------------------------------------
# -------------------------------------------------------------------------
if __name__ == "__main__":
    xml_file = "data/OMX-OTF-683nm-2d.xml"          # ← path to the file you showed above
    otf_cube = load_fairsim_otf(xml_file)   # (bands, 512, 512)
    otf = otf_cube[2]  # use the first band
    otf = np.fft.fftshift(otf)  # shift low frequencies to the centre              
    plt.imshow(np.abs(otf), cmap="gray")
    plt.colorbar()
    plt.title("OTF magnitude")
    plt.show()
    print("OTF shape:", otf.shape)   # (512, 512)
    print("Centre value (0 freq):", otf[256, 256])
