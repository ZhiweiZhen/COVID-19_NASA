# COVID-19_NASA
This is a Pytorch implementation of Geometric Deep Learning Models for the task to uncover hidden patterns in Covid-19 with  integrating NASA’s satellite observations.

# Requirement 
pyTorch <br />
tqdm <br />
numpy <br />
networkx 


# Training

To run the experiment, set the dictionary to folder "CODE" and create the folder "OUTPUT".
```
python3 <model_name>.py 
```

All experiment results can be found in the folder "OUTPUT"

# DATA

Covid-19:<br />
Daily records on COVID-19 and hospitalizations are from CovidActNow project. We follow the guidelines from the Models of Infectious Disease Agent Study (MIDAS) for COVID-19 modeling research. We use daily time series data from February 1 to December 31, 2020. Hospitalization numbers were publicly available for all three states, at a county level.

NASA dataset:<br />
NASA’s Distributed Active Archive Centers (DAAC) servers provide publicly available access to the original datasets through the products: 1) Atmospheric Infrared Sounder (AIR)/Aqua L3 Daily Standard Physical Retrieval (AIRS3STD) for temperature and relative humidity, and 2) the Atmosphere Daily Global Product from Moderate-resolution Imaging Spectroradiometers (MODIS) on Terra(MOD08 D3)  for AOD.

We take ime series data from February 1 to December 31, 2020. February 1 to December 31, 2020.
AOD dataset is calculated using AOD at 550 nm wavelength. We match each county-level time series with its corresponding Federal Information Processing Standard Publication 6-4 (FIPS 6-4)code. 

Socioeconomic variables:<br />
We generate a socioeconomic-based network for each state under study using 9 socioeconomic variables. A socioeconomic matrix distance, via the euclidean metric, is usde to build  county-level connections which serve as input to each GNN model. Five variables come from the Centers for Disease Control and Prevention (CDC)/Agency for Toxic Substances and Disease Registry (ATSDR) Social Vulnerability Index: Socioeconomic Status, Household Composition & Disability, Minority Status & Language, Housing Type & Transportation, and Overall Vulnerability Index.The rest of the variables are part of The COVID-19 Vac- cine Coverage Index (CVAC): Historic Undervaccination, Sociodemographic Barriers, Resource-Constrained Healthcare System, and Healthcare Accessibility Barriers.


# Cite
