# Susceptibility Zoning plugin (SZ)

**! This README file is a draft and is currently being improved**

## Introduction

This repository contains the code for a plugin for [QGIS](https://www.qgis.org), called "Susceptibility Zoning plugin", aiming at creating maps of susceptibility to various natural forcing elements.

The plugin has been developed with a main focus and application towards landslides susceptibility, but it can be applied to different types of natural events or inter-disciplinary applications.

The plugin uses a bi-variate "Weight of Evidence" (WoE) model and the Frequency Ratio (FR) as first statistical methods to evaluate the susceptibility of a study area to specific event. Additional methods (_other examples to be added_) are being implemented and will be added to the plugin as soon they are ready.

## How it works


### Weight of Evidence and Frequency Ratio

_Add a short description of WoE and a few useful reference links_

The WoE is a bi-variate statistical method used for classification. It was introduced by by Agterberg et al. (1989) and then by Bonham-Carter et al. [(1988)](https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/SC010p0015) for spatial analysis. The model, evaluate the predictive power of an independent variable (cause) in relation to the dependent variable (in our study landslide) by the assignment of two weights (W+, W-).

The positive weight defines that the independent variable is favorable to landslide occurrence; on the contrary the negative ones. The sum of W+ and W- and of all the independent variables considered provides the Susceptibility Index (SI).

W+ = ln((Npx1/(Npx1+Npx2))/(Npx3/(Npx3+Npx4)))

W- = ln((Npx2/(Npx1+Npx2))/(Npx4/(Npx3+Npx4)))

SI = Σ(W+ - W-)

Npx1 is the number of pixels representing the presence of both independent variable and dependent variable; Npx2 is the number of pixels representing the presence dependent variable and absence of independent variable; Npx3 is the number of pixels representing the presence of independent variable and absence of dependent variable; Npx4 is the number of pixels representing the absence of both independent variable and dependent variable [(Dahal et al., 2008)](https://link.springer.com/article/10.1007/s00254-007-0818-3)

As the WoE, the Frequency Ratio (FR) is a simple bi-variate statistical method often used for classification.

FR = (Npx1/Npx2)/(ΣNpx1/ΣNpx2)

SI = ΣFR

Npx1 = The number of pixels containing the dependent variable in a class; Npx2 = Total number of pixels of each class in the whole area;
ΣNpx1 = Total number of pixels containing the event; ΣNpx2 = Total number of pixels in the study area

### Input data

### Classification of zones

## The plugin interface

<img src="https://github.com/CNR-IRPI-Padova/SZ/blob/master/images/Screenshot1.png" width="500">

_Fig. 1 Input causes from the main GUI section of SZ plugin for susceptibility mapping._

<img src="https://github.com/CNR-IRPI-Padova/SZ/blob/master/images/Screenshot2.png" width="500">

_Fig. 2 Rest of input from the main GUI section of SZ plugin for Susceptibility mapping._

<img src="https://github.com/CNR-IRPI-Padova/SZ/blob/master/images/Screenshot3.png" width="500">

_Fig. 3 Input data for Susceptibility Index ROC based classification and validation._

## Applications

_A few examples and references about Belt and Road Initiative_

## Presentations

_A list of presentations made about the plugin and its applications_

## Collaborations

...
