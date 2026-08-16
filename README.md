# MKBS CASA0010 Dissertation
This repository contains the code that has been used to utilise BODS GTFS-RT data to understand actual bus performance in Bristol, Leeds and Manchester on four Wednesdays (3 June, 10 June, 17 June, 24 June) in June 2026 and explore hypothetical improvement scenarios. This is for the purpose of completing CASA0010 Dissertation requirements as part of the MSc Urban Spatial Science programme at the Centre for Advanced Spatial Analytics (CASA) at University College London (UCL).

The Python scripts are all in the `Code` folder. Before running the scripts, please ensure that the following packages have been installed.
- `rt2gtfs` - this is a new package developed by [Chen and Botta, 2026](https://arxiv.org/html/2603.11477v2) specifically to convert BODS GTFS-RT into a static GTFS format
- `r5py` - this requires Java Development Kit to also be present in your terminal, thus may require slightly extended troubleshooting before this package could work

Please also ensure that all the datasets listed in `1_initial.ipynb` are ready in the `Data\Preprocessed` subfolder.

Simply run the scripts in order from 1 to 5 in order to obtain the results. Feel free to edit the code to model other cities or to incorporate alternative methodologies!