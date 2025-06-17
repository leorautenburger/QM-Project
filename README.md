# QM-Project
UCL Quantitative Methods 1 Final Project

This is an archive of my final project for Quantitative Methods 1 at UCL. The project is a time-series analysis of stock returns among the current Fortune 25 against their google trends index.

The analysis was conducted using a fixed-effects panel regression framework implemented in python using statsmodels, pandas and numpy. Analysis was conducted generally and then sector-by-sector, various levels of quarterly lag were also introduced and tested.

Data for stock market performance was sourced from Yahoo Finance, and Google Trends data was sourced from Google Trends.

The full write-up can be found in 'write-up.pdf', which includes more details on specifics of methodology, literature review, and the results of the analysis.

Code used to combine and clean stock market and Google Trends data can be found in 'Stock CSV Stitcher.py' and 'Trends CSV Stitcher.py' respectively.

Code used to perform sector and general analysis, and to plot results, can be found in 'General Analysis.py' and 'Sector Analysis.py' respectively. These files can be modified easily to vary lag etc.
