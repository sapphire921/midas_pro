# midas_pro
Python version of Mixed Data Sampling (MIDAS) regression (allow for multivariate MIDAS)

This package is developed based on midaspy. This version can be used for MIDAS regression and multivariate MIDAS regression.

**A brief introduction to MIDAS model:**

Mixed-data sampling (MIDAS) model is a direct forecasting tool which can relate future low-frequency data with current and lagged high-frequency indicators, and yield different forecasting models for each forecast horizon. It can flexibly deal with data sampled at different frequencies and provide a direct forecast of the low-frequency variable. It incorporates each individual high-frequency data in the regression, which solves the problems of losing potentially useful information and including mis-specification.

MIDAS model can have more than one high-frequency indicator at the same time which lead to the Multivariate-MIDAS (Multi-MIDAS) model. The high-frequency indicators considered can have different theta parameters, different sampling frequencies and different lag length.
