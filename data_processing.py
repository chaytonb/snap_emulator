import numpy as np
from datetime import datetime, timedelta
import xarray as xr
import os

def process_t6_data(output_file_path, target_var):
    """Process one snap run and save as npz file"""

    # Start date
    start_date = datetime(2023, 1, 1)
    # End date
    end_date = datetime(2023, 12, 31)

    # First 6 hours left as spinup
    spinup_hours = 6

    forecast_times = ['00', '06', '12', '18']

    # initialise numpy arrays
    preds_array = None
    target_array = None

    # Iterate over each day
    current_date = start_date
    while current_date <= end_date:
        try:
            day = current_date.strftime("%d")
            month = current_date.strftime("%m")
            year = current_date.strftime("%y")

            for time in forecast_times:
                # Load meteo data
                meteo_path = f'/lustre/storeB/immutable/archive/projects/metproduction/MEPS/2023/{month}/{day}/meps_det_2_5km_2023{month}{day}T{time}Z.nc'
                meteo_data = xr.open_dataset(meteo_path, engine="netcdf4")
                meteo_data = meteo_data[['x_wind_10m', 'y_wind_10m', 'surface_air_pressure']]
                meteo_data = meteo_data.isel(x=slice(188, 524), y=slice(97, 433)) # Dimensions: 336x336
                
                # Load SNAP data
                snap_file_path = f'output_files/ringhals_2023{month}{day}_{time}Z.nc'
                snap_data = xr.open_dataset(snap_file_path, engine="netcdf4")
                snap_data = snap_data[[target_var, 'instant_height_boundary_layer']]
                snap_data = snap_data.isel(x=slice(188, 524), y=slice(97, 433)) # Dimensions: 336x336

                # Log transform concentration values
                concentration_values = snap_data[target_var].values
                # Create an array for the result
                log_conc = np.zeros_like(concentration_values)
                # Apply log transformation to values that are > 0
                log_conc[concentration_values > 0] = np.log(concentration_values[concentration_values > 0])
                # Set values smaller than 1e-20 to 0
                log_conc = np.where(log_conc < -20, 0, log_conc)
                # Add 20 to all non-zero values
                log_conc = np.where(log_conc != 0, log_conc+20, 0)
                # Assign the result back to the original data structure
                snap_data[target_var] = (('time', 'y', 'x'), log_conc)
                
                preds_data = np.stack([
                    meteo_data['x_wind_10m'].isel(time=spinup_hours+1, height7=0).values,
                    meteo_data['x_wind_10m'].isel(time=13, height7=0).values, # T+6
                    meteo_data['y_wind_10m'].isel(time=spinup_hours+1, height7=0).values,
                    meteo_data['y_wind_10m'].isel(time=13, height7=0).values, # T+6
                    snap_data['instant_height_boundary_layer'].isel(time=spinup_hours).values,
                    snap_data['instant_height_boundary_layer'].isel(time=12).values, # T+6
                    meteo_data['surface_air_pressure'].isel(time=spinup_hours+1, height0=0).values,
                    meteo_data['surface_air_pressure'].isel(time=13, height0=0).values # T+6
                    ], axis=0)[np.newaxis, :, :, :] # Add new axis run number

                target_data = snap_data[target_var].isel(time=-1).values[np.newaxis, :, :] # Take final time step, add new axis

                # Check if the file exists
                if preds_array is not None: 
                    preds_array = np.concatenate((preds_array, preds_data), axis=0)
                else:
                    preds_array = preds_data

                if target_array is not None:
                    target_array = np.concatenate((target_array, target_data), axis=0)
                else:
                    target_array = target_data
                
        except Exception as e:
            print(f"Error processing {current_date}: {str(e)}")

        current_date += timedelta(days=1)

    # Save the combined arrays into npz files
    np.savez_compressed(output_file_path, pred_vars=preds_array, target_vars=target_array)
    
    return current_date


if __name__ == "__main__":

    directory = 'combined_t6_data_accumulated.npz'
    target_var = 'PMCH_acc_concentration'

    current_date = process_t6_data(directory, target_var)
