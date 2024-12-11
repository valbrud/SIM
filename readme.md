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
## 1. Change figure and animation paths to whatever you need in the file `globvar.py`
## 2. Go to the directory `article_plots` and run the corresponding tests in the file `test_article_plots.py` in your IDE or with
```bash
python -m unittest article_plots.plot_ssnr_article.TestArticlePlots.test_ring_averaged_ssnr 
```

# To access the documentation open the file `docs/index.html` in your browser
 The Documentation is generated with the help of the Sphinx package. You may need to install additional dependencies to work with it.   
 You can quick check if the installation is successful with   
`python -m input_parser test_illumination.conf  --gui`  
 That should launch the user interface for the interference patterns plotting

# Contact information
Feel free to address any questions about the code or related articles to:
- Author(s): Valerii Brudanin
- Email: v.s.brudanin@tudelft.nl
- Institution: Delft University of Technology, Applied Sciences, Imaging Physics Department