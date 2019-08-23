# Hapke
This folder contains codes to optimize parameters involved in Hapke's theory which happens to be a standard principle aimed at determining the abundance of minerals in lunar land using reflectance spectra of the surface. The project was part of the Planatary Science deparment, SAC, ISRO. 

# Objective
Retrieval of model parameters for non-linear spectral unmixing of lunar mineral end members using different optimization techniques.

# Work Description
The objective was to find parameters involved in Hapke’s radiative transfer theory. Hapke’s theory is
used to determine the reflectance of a given lunar land’s patch containing different end members in
some proportion using an equation. Using this theory we can determine the proportion of minerals in
a given land. This equation contains some parameters which depends on the surface of the lunar body.
We used Lunar Soil Characterization Consortium (LSCC) dataset for ground truth. Using Nelder-Mead
optimization method we found value of parameters w,h,b and c involved in Hapke’s theory

# Tools and Libraries:
* Pandas
* Numpy
* lmfit
