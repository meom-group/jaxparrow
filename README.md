# Cyclogeostrophic Balance Code

This repository contains code for performing inversion of oceanographic data using various algorithms and approaches. The goal of the code is to estimate the velocity of an ocean system from sea surface height observations.

To use this code, you will need to install the following dependencies:

* JAX
* Xarray
* Numpy.ma module

## Structure

The code is structured as follows:

* **Model**: Contains the implementation of "Model" object that is responsible for storing all the data taken from the netCDF files.
* **PreProcessor**: Contains the implementation of the "PreProcessor" object that is responsible for processing the data from the netCDF files and storing them in the model object. The code works with a C-type grid and requires four files: sea surface height (ssh), u-component, v-component, and mask files. Depending on the configuration of the files, the code may need to be modified to read the data correctly.
* **Processor**: Contains the implementation of the "Processor" object that is responsible for performing the inversion of the data and storing the results in the model object.
* **PostProcessor**: Contains the implementation of the "PostProcessor" object that is responsible for exporting the inversion results to netCDF files. The output consists of two files: u and v components of the cyclogeostrophic velocity. The files will be saved in the same folder as the input files.

## Usage

To use the inversion code, you need to provide input data in the form of netCDF files, as well as a configuration file specifying the inversion parameters.

The inversion code implements algorithms for estimating the cyclogeostrophic velocity from the data, including:

* A new gradient-based approach
* Iterative method

Before using the code, it is important to verify the path to the netCDF files and their names. These parameters can be changed in "PreProcessor.py" under the variables: "dir_data", "name_mask", "name_ssh", "name_u", and "name_v". Alternatively, you can uncomment the lines that call a function allowing interactive selection of the files (first line in the functions "read_ssh", "read_u", "read_v", and "read_mask") and comment the manual ones.

The default method to solving the problem is the new gradient-based approach. To use the iterative method, you need to change the variable "method" in "Processor.py", function "solve_model", to "1".

The files containing the data used during the implementation of the code can be found using the following link:
https://1drv.ms/f/s!Aq7KsFIdmDGepjMT6o77ko-JRRZu?e=hpxeKa

This repository also contains a notebook demonstrating the basic functionalities of the code.

## License

Copyright 2023. See [LICENSE](https://github.com/VictorZaia/cyclogeostrophic_balance/blob/main/LICENSE) for additional details.
