### Environmental Justice and Lessons Learned from COVID-19 Outcomes – Uncovering Hidden Patterns with Geometric Deep Learning and New NASA Satellite Data
[Abstract] Virtually all aspects of our societal functioning -- from food security to energy supply to healthcare -- depend on the dynamics of environmental factors.
Nevertheless, the social dimensions of weather and climate are noticeably less explored by the artificial intelligence community. By harnessing the strength of geometric deep learning (GDL), we aim to investigate the pressing societal question the potential disproportional impacts of air quality on COVID-19 clinical severity. To quantify air pollution, here we use aerosol optical depth (AOD) records which measure the reduction of the sunlight due to atmospheric haze, dust, and smoke. We also introduce unique and not yet broadly available NASA satellite records (NASAdat) on AOD, temperature and relative humidity and discuss the utility of these new data for biosurveillance and climate justice applications, with a specific focus on COVID-19 in the States of Texas and Pennsylvania in USA. The results indicate that, in general, the poorer air quality tends to be associated with higher rates for clinical severity and, in case of Texas, this phenomenon particularly stands out in Texan counties characterized by higher socioeconomic vulnerability. This, in turn, raises a concern of environmental injustice in these socio-economically disadvantaged communities. Furthermore, given that one of NASA's recent long-term commitments is to address such inequitable burden of environmental harm by expanding use of Earth science data such as NASAdat, this project is one of the first steps toward developing a new platform integrating NASA's satellite observations with DL tools for social good.

### Requirements
Please check the Requirements.txt.

### Test an Example
To replicate the results, please set the dictionary to the folder "CODE" and create the folder "OUTPUT". All experiment results can be found in the folder "OUTPUT".
```
python3 <model_name>.py 
```

### Datasets

#### COVID-19
Daily records on COVID-19 and hospitalizations are from CovidActNow project. We follow the guidelines from the Models of Infectious Disease Agent Study (MIDAS) for COVID-19 modeling research. Hospitalization numbers were publicly available for all three states, at a county level.

#### NASA Data
NASA’s Distributed Active Archive Centers (DAAC) servers provide publicly available access to the original datasets through the products: (1) Atmospheric Infrared Sounder (AIR)/Aqua L3 Daily Standard Physical Retrieval (AIRS3STD) for [temperature](https://commons.datacite.org/doi.org/10.48577/jpl.z31y-2r10) and [relative humidity](https://commons.datacite.org/doi.org/10.48577/jpl.ws86-1q81), and (2) the [Atmosphere Daily Global Product](https://commons.datacite.org/doi.org/10.48577/jpl.k37v-y751) from Moderate-resolution Imaging Spectroradiometers (MODIS) on Terra (MOD08 D3) for AOD. [AOD dataset](http://dx.doi.org/10.5067/MODIS/MOD08_M3.006) is calculated using AOD at 550 nm wavelength. We match each county-level time series with its corresponding Federal Information Processing (FIP) Standard Publication 6-4 (FIPS 6-4) code. 

#### Socioeconomic Variables
We generate a socioeconomic-based network for each state under study using 9 socioeconomic variables. A socioeconomic matrix distance, via the euclidean metric, is used to build county-level connections which serve as input to each model. Five variables from the Centers for Disease Control and Prevention (CDC)/Agency for Toxic Substances and Disease Registry (ATSDR) Social Vulnerability Index: Socioeconomic Status, Household Composition & Disability, Minority Status & Language, Housing Type & Transportation, and Overall Vulnerability Index.The rest of the variables are part of The COVID-19 Vaccine Coverage Index (CVAC): Historic Undervaccination, Sociodemographic Barriers, Resource-Constrained Healthcare System, and Healthcare Accessibility Barriers.

#### References
