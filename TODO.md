- Improve GUI functionality, 
  - Fix configuration/illumination inconveniences
  - Implement change source button functionality
  - Append the documentation
  - Separate computational and rendering threads to avoid freezes
- Write some test for numeric consistency checking (not image-based)
- Improve OpticalSystems module 
  - Implement confocal and ISM classes. 
  - Add natural interface for pupil function
  - Rewrite PSF computations design
- Implement the full SIM reconstruction pipeline
  - Implement the ProcessorSIM class as a top-level manager for all SIM classes
  - Make possible downloading data from databases
  - Implement the estimation of the illumination parameters from data
  - Implement regularization and apodization filters
  - Implement mutual information based object estimation
  - Fix existent errors and inconsistencies
- Optimize code performance 
  - Change slow numpy.fft to pyfftw or scipy.fft
  - Profile the code
  - Implement bottlenecks in c++
  - Move image related operations on CUDA
  - Allow work with several images in parallel through MPI
- Increase user-friendliness
  - Switch user interface to web?
  - Draw UML diagrams?
