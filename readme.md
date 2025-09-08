# Installation
### 1. Clone the repository
```bash
git clone https://github.com/valbrud/SIM.git
```
### 2. Install the dependencies
```bash
pip install -r requirements.txt
```
If pip complaints about python versioning, ignore it and install missing packages manually with pip install 'package_name'. 

### 3. (Optional) Deploy in the container
```bash
docker build -t sim .
```
Alternatively, if you work in IDE, it will probably propose you to create a virtual environment for the project. In any case, I consider version problems with the packages in use to be rather low. 

### 4. Check that installation is successful   
You can most easily check that the installation is successful and (most of) the functionality there by navigating to the `Box`graphic module in the folder GUI and launching it with

`python -m input_parser C5C.conf  --gui -ip`  

 That should launch the user interface for the interference patterns plotting

# GUI
### There are currently following graphic modules present: 
- Box

### BOX
This module allows to analyze easily the interference of electromagnetic plane waves. It provides functionality to load and modify plane waves configuration, as well as to work the intensity patterns they produce and their Fourier transforms.


# Documentation
### To access human-readable project doumentation, open the file `doc/_build/html/index.html` in your browser
 The Documentation is generated with the help of the Sphinx package. You may need to install additional dependencies to work with it. You can regenerate the documentation by going to the `doc` directory and `make html`

# Reproducibility
### To reproduce the results from one of the manuscripts below:
### 1. Go to the directory `papers` and navigate to the corresponding folder
### 2. Launch the test of interest. Tests roughly correspond to the figures in the paper. If your IDE doesn't recognize a test with a right-click, use the following syntax 
```bash
python -m unittest papers.ssnr_comparison.generate_plots.TestArticlePlots.test_ring_averaged_ssnr 
```
I try to maintain complete back-compatibility of newer versions. However,  you can also launch the very same code that was used during the release of the paper by downloading an appropriate version of the project from GitHub.

# Manuscripts
- "Comparison of 3D structured illumination microscopy configurations in terms of spectral signal to noise ratio", OpticsExpress, 2025, https://doi.org/10.1364/OE.553750

Folder: `ssnr_comparison` 

Project version: 1.0.0

# Contact information
Feel free to address any questions about the code or related articles to:
- Author(s): Valerii Brudanin
- Email: v.s.brudanin@tudelft.nl
- Institution: Delft University of Technology, Applied Sciences, Imaging Physics Department