# Data Utilities


The raw and labelled data are contained in large segy files and need to be pre-processed into one or more npy files for use by the training scripts. 


## process_all_files.py

THe `process_all_files.py` script can work with SEG-Y files and output data on both blob storage and local disk. This script will process segy files regardless of their structure and output npy files for use in training/scoring. If the files contain seismic data, the data will also be normalized to have values between 0 and 1.

The resulting npy files will use the following naming convention:

```<prefix>_<inline id>_<xline id>_<depth>.npy```

These inline and xline ids are the upper left location of the data contained in the file and can be later used to identify where the npy file is located in the segy data.

To use this script, follow these examples:

1) Process all SEG-Y data on blob storage, generate single npy file per SEG-Y and upload results back to same blob storage account.

    ```
    python process_all_files.py --account_name $ACCOUNT_NAME --account_key $ACCOUNT_KEY --tmp_dir /tmp --cube_size -1 
            --input_path seismicdata/input --output_path seismicdata/cubes
    ```

2) Process single SEG-Y file on blob storage, generate single npy file per SEG-Y and upload results back to blob storage.

    ```
    python process_all_files.py --account_name $ACCOUNT_NAME --account_key $ACCOUNT_KEY --tmp_dir /tmp --cube_size -1
            --segy_file listric1_input.segy --input_path seismicdata/input --output_path seismicdata/cubes
    ```

3) Process single SEG-Y file on blob storage, generate npy files of shape (128,128,128) and upload results back to blob storage.
python

    ```
    python process_all_files.py --account_name $ACCOUNT_NAME --account_key $ACCOUNT_KEY --tmp_dir /tmp --cube_size 128
            --segy_file listric1_input.segy --input_path seismicdata/input --output_path seismicdata/cubes
    ```

4) Process a local SEG-Y file in the /mnt/data/segy directory, generate npy files of shape (128,128,128) and keep results locally.

    ```
    python process_all_files.py --account_name $ACCOUNT_NAME --account_key $ACCOUNT_KEY --tmp_dir /tmp --cube_size 128
            --segy_file listric1_input.segy --input_path segy --output_path cubes --local_output --local_input  
            --input /mnt/data --ouput /mnt/data
    ```


There are several additional command line arguments that may be needed to load specific segy files (i.e. the byte locations for data headers may be different). Run --help to review the additional commands if needed.


# Data normalization

Usually, seismic data has a large range of amplitude, such as from -66858.68 to 68263.31 or from -9.64 to 9.75, because these values represent measure of subsurface reflectivity.
In order to use this type of data for training a neural network, we must to put it within a reasonable range. 
Currently we are normalizing seismic data to be within [0, 1].

## Testing

Run pytest from the segyconverter directory