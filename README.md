Virtual Metabolomics Mass Spectrometry (ViMMS)
==============================================

Abstract
--------

Liquid-Chromatography (LC) coupled to tandem mass spectrometry (MS/MS) is widely used in identifying small molecules in
untargeted metabolomics. Various strategies exist to acquire MS/MS fragmentation spectra; however, the development of 
new acquisition strategies is hampered by the lack of simulators that let researchers prototype, compare, and optimise 
strategies before validations on real machines. We introduce **V**irtual **M**etabolomics **M**ass **S**pectrometer 
(**VIMMS**), a modular metabolomics LC-MS/MS simulator framework that allows for scan-level control of the MS2 acquisition 
process *in-silico*. ViMMS can generate new LC-MS/MS data based on empirical data or virtually re-run a previous LC-MS/MS 
analysis using pre-existing data *in-silico* to allow the testing of different fragmentation strategies. It allows the 
comparison of different fragmentation strategies on real data, with the resulting scan results extractable as mzML files. 

To demonstrate its utility, we show how our proposed framework can be used to take the output of a real tandem mass 
spectrometry analysis and examine the effect of varying parameters in Top-N Data Dependent Acquisition protocol. 
We also demonstrate how ViMMS can be used to compare a recently published Data-set-Dependent Acquisition strategy with 
a standard Top-N strategy. We expect that ViMMS will save method development time by allowing for offline evaluation of 
novel fragmentation strategies and optimisation of the fragmentation strategy for a particular experiment.

Installation
------------

1. Install Python 3. We recommend Python 3.6 or 3.7.
2. Install pipenv (https://pipenv.readthedocs.io/en/latest/).
3. In this directory, run `$ pipenv install` to create a new virtual environment and install all the packages we need.
4. Run jupyter notebook. 

Example Notebooks
--------

Examples are available in the `examples` folder.