## Test data used in the example workflow notebooks

Test data used in the example workflow notebooks are sea surface temperature (SST) over the Pacific and clustered 2 meter temperature (T2M) over the North America. The fields used here are processed outputs from original ERA5 dataset.

## Reproduce test data

To prepare the test data yourself, first you need to download the SST and T2M fields of ERA5 dataset from CDS. It is recommended to download these fields using [`era5cli`](https://era5cli.readthedocs.io/en/stable/). For instance, for the T2M field, you can collect the data via the following command:

```sh
era5cli hourly --variables 2m_temperature --startyear 1959 --endyear 2021 --area 70 225 30 300
```

It is the same for the SST, but with different area (`--area 50 175 25 240`).

Note that your query to CDS and the download could take relatively long. In total, you need to download about 50GB data.

When you have all the required data, you can perform upscale sampling of your data using [`cdo`](https://code.mpimet.mpg.de/projects/cdo/). For example, to upscale the data to daily timescale, you can run the following command:

```sh
cdo -b 32 settime,00:00 -daymean -mergetime /path_to_all_your_data /path_to_your_output_dir/t2m_1959-2021_1_12_daily_025deg.nc
```

Now we can regrid it to 2 degree resolution, using the following command:

```sh
cdo remapbil,r180x90 t2m_1959-2021_1_12_daily_025deg.nc t2m_1959-2021_1_12_daily_2deg.nc
```

It is the same for the SST, but with even coarser resolution (5 degree).

Now the SST field is ready, we still need to follow this [notebook](./prepare_test_data.ipynb) to prepare the clustered T2M data.

## License
Data used here is generated using Copernicus Climate Change Service information and for more information about licensing, please check the [Licence Agreement](https://cds.climate.copernicus.eu/cdsapp/#!/terms/licence-to-use-copernicus-products) for Copernicus Products.