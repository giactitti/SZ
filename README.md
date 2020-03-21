# Susceptibility Zoning plugin (SZ)

**! This README file is a draft and is currently being improved**

## Introduction

This repository contains the code for a plugin for [QGIS](https://www.qgis.org), called "Susceptibility Zoning plugin", aiming at creating maps of susceptibility to various natural forcing elements.

The plugin has been developed with a main focus and application towards landslides susceptibility, but it can be applied to different types of natural events or inter-disciplinary applications.

The plugin uses a bi-variate "Weight of Evidence" (WoE) model and the Frequency Ratio (FR) as first statistical methods to evaluate the susceptibility of a study area to specific event. Additional methods (_other examples to be added_) are being implemented and will be added to the plugin as soon they are ready.

## How it works


### Weight of Evidence and Frequency Ratio

_Add a short description of WoE and a few useful reference links_

The WoE is a bi-variate statistical method used for classification. It was introduced by by Agterberg et al. (1989) and then by Bonham-Carter et al. [(1988)](https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/SC010p0015) for spatial analysis. The model, evaluate the predictive power of an independent variable in relation to the dependent variable by the assignment of two weights (W+, W-).

The positive weight defines that the independent variable is favorable to landslide occurrence; on the contrary the negative ones. The sum of W+ and W- and of all the independent variables considered provides the Susceptibility Index.

As the WoE, the Frequency Ratio (FR) is simple a bi-variate statistical method used for classification.

### Input data

### Classification of zones

## The plugin interface

## Applications

_A few examples and references about Belt and Road Initiative_

## Presentations

_A list of presentations made about the plugin and its applications_

## Collaborations

...
