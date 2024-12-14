# To setup the project, follow these steps:
## 1. Clone the repository
```bash
git clone https://github.com/valbrud/SIM.git
```
## 2. Install the dependencies
```bash
pip install -r requirements.txt
```
## 3. (Optional) Deploy in the container
```bash
docker build -t sim .
```

At the current stage, the project is fully written in Python, thus no complications are expected 
so far the required libraries are downloaded. 

# To reproduce the results from the article
## Go to the directory `papers/ssnr_comparison` and run the corresponding test in the file `generate_plots.py` in your IDE or with
```bash
python -m unittest papers.comparison_ssnr.generate_plots.TestArticlePlots.test_ring_averaged_ssnr 
```
If you want to run other unittests from the unittests folder, you may need to change paths defined in the file globvar.py

# To access the documentation open the file `docs/index.html` in your browser
 The Documentation is generated with the help of the Sphinx package. You may need to install additional dependencies to work with it.   
 You can quick check if the installation is successful with   
`python -m input_parser five_s_waves.conf  --gui -ip`  
 That should launch the user interface for the interference patterns plotting

# GUI
Gui is not fully implemented yet. Now it can only be used to plot interference patterns of configurations in the 
corresponding configuration files (config/*.conf). They can be uploaded with a button Load Configuration. 
Load Illumination and some other buttons will result in a crash. You can, however, iterate through slices with the slider
and change the plane of view (xy, yz, xz) with a change view button. 

# Contact information
Feel free to address any questions about the code or related articles to:
- Author(s): Valerii Brudanin
- Email: v.s.brudanin@tudelft.nl
- Institution: Delft University of Technology, Applied Sciences, Imaging Physics Department