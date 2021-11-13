# Susceptibility Zoning plugin (SZ)

## Introduction

This repository contains the code for a plugin for [QGIS](https://www.qgis.org), called "Susceptibility Zoning plugin" (SZ-plugin), aiming at creating maps of susceptibility to various natural forcing elements.

The plugin has been developed with a main focus and application towards landslides susceptibility, but it can be applied to different types of natural events or inter-disciplinary applications.

The plugin uses several type of statistical model for susceptibility evaluation, such as:

* Weight of Evidence
* Frequency Ratio
* Logistic Regression
* Decision Tree
* Support Vector Machine
* Random Forest

The plugin allows to cross-validate the results by simple random selection of test/train samples or allows to cross-validate by k-fold cross-validation method.


## Installation

The SZ plugin is not an official QGIS plugin.

It can be installed on QGIS3.x cloning the repository or downloading it as zip file (and than unzipping it) and copying the _sz_module_ folder in your local python/plugin folder (read [here](https://docs.qgis.org/3.10/en/docs/user_manual/plugins/plugins.html#core-and-external-plugins) for more information).

<img src="./images/install.png" width="500">
<p>
  
At the end you should have the SZ plugin in your processing toolbox

<img src="./images/gui.png" width="500">
<p>

Then you need to install the basic dependencies to run the project on your system:

```
cd sz
pip install -r requirements.txt
```
or you can install them separately
  
### GUI

The functions are grouped into 3 cathegories:
* _Data preparation_
* _SI_
* _SI k-fold_
* _Classify SI_

_Data preparation_ functions can be used for data pre-processing
_SI_ functions run the statistic models for susceptibility, cross-validate by a simple random selection of train/test samples and evaluate the prediction capacity by ROC curves
_SI k-fold_ functions run the statistic models for susceptibility, cross-validate by k-fold method and evaluate the prediction capacity by ROC curves
_Classify SI_ functions allows to cathegorize the susceptibility index into _n_ classes on the base of AUC maximization.

### Input data of SI and SI k-fold functions

Input data for SI k-fold or SI functions should be a vector layer with a number of fields for independet variables ans a field for the dependent variable classified binomially: 0 for absence, >0 for presence.

<img src="./images/use.png" width="500">
 
### Test

A dataset and QGIS project are available in .... to test the plugin.

<img src="./images/test.png" width="500"> 

<img src="./images/output.png" width="500">


## Third-part libraries and plugins used

* [GDAL](https://gdal.org/)
* [Scikit-learn](https://scikit-learn.org/stable/index.html)
* [Matplotlib](https://matplotlib.org/)
* [Plotly](https://plotly.com/)
* [Pandas](https://pandas.pydata.org/)



## Applications

_A few examples and references about applications_

Titti, G., Borgatti, L., Zou, Q., Pasuto, A., 2019. Small-Scale landslide Susceptibility Assessment. The Case Study of the Southern Asia. Proceedings 30, 14. [10.3390/proceedings2019030014](https://doi.org/10.3390/proceedings2019030014)

## Presentations

_A list of presentations made about the plugin and its applications_

Titti, Giacomo, Sarretta, Alessandro, Crema, Stefano, Pasuto, Alessandro, & Borgatti, Lisa. (2020, March). Sviluppo e applicazione del plugin Susceptibility zoning per il supporto alla pianificazione territoriale. Zenodo. [10.5281/zenodo.3723353](https://zenodo.org/record/3723353)

## Credits

Giacomo Titti and Alessandro Sarretta, Padova, March 2020

please cite as: Giacomo Titti, & Alessandro Sarretta. (2020, May 25). CNR-IRPI-Padova/SZ: SZ plugin (Version v0.1). Zenodo. http://doi.org/10.5281/zenodo.3843275
