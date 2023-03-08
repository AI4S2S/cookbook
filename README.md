# cookbook
Your recipe for building and executing your workflow with s2spy suite.

## Tasks and missions
- Finish the entire workflow with s2spy
  - Apply basic workflow to CMIP6 data
  - Apply basic workflow to EC46 data
- Make a pipeline?

## New features to expect
- Multiple lags within RGDR
- Double cross-validation to the workflow
- Implement the linear algebra type of dimensionality reduction e.g. FT (and contribute to `xeofs`?).
- Parallel computing and memory usage optimization with `Dask` and `Zarr`
- Create calendar object compatible preprocessor
- New preprocessors that can deal with NaNs in different ways (e.g. mean, interpolation, etc.)

If the preprocessor becomes more comprehensive and independent, we may want to split it off, just like `lilio`.

## Basic workflow
Here is an example of a basic workflow with s2spy suite.

1. Define calendar
2. Input data: SST and soil moisture from ERA5 <br>
Target: US temperature from ERA5 (see notebook from `scratch`)
3.	Map the calendar to the data
4.	Train-test split based on the anchor years-> 70%/30% split (outer cv loop)
5.	Mask the data to get only full training years
6.	Fit (out of sample) preprocessing (incl. detrend, remove climatology, rolling mean) to the masked data
7.	Preprocess all data
8.	Resample all data to the calendar
9.	Train-test split based on the previous split (outer cv loop -> inner cv loop)
10.	Dimensionality reduction
11.	Fit the ML model (Ridge) and transform to the test data
12.	Evaluate the results (skill metrics, visualization) and workflow (time and memory usage)