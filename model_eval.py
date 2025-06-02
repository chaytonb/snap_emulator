import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
import xarray as xr
import random
import unet
from tqdm import tqdm
import torch.nn.functional as F
from scipy.stats import pearsonr
from config import DATA_MODE, DATA_PATHS

def get_projection(ds):
    attr = None
    if "crs" in ds.attrs:
        attr = ds.attrs["crs"]
    if "proj4" in ds.attrs:
        attr = ds.attrs["proj4"]
    if attr:
        proj_string = attr.removeprefix("+init=")
        if proj_string == "epsg:32633":
            return ccrs.UTM(33)
        elif proj_string == "+proj=longlat":
            return ccrs.PlateCarree()
        return ccrs.Projection(proj_string)
    if "spatial_ref" in ds:
        var = ds["spatial_ref"]
        if var.attrs["projected_crs_name"] == "ETRS89 / UTM zone 33N (N-E)":
            return ccrs.UTM(33)
        elif var.attrs["projected_crs_name"] == "ETRS89 / UTM zone 32N (N-E)":
            return ccrs.UTM(32)
        else:
            raise NotImplementedError()
    if "projection_lambert" in ds.variables:
        pr = ds["projection_lambert"].attrs
    elif "projection_laea" in ds.variables:
        pr = ds["projection_laea"].attrs
    else:
        pr = ds["projection"].attrs
    if pr["grid_mapping_name"] == "rotated_latitude_longitude":
        projection = ccrs.RotatedPole(
            pole_longitude=pr["grid_north_pole_longitude"],
            pole_latitude=pr["grid_north_pole_latitude"],
        )
    elif pr["grid_mapping_name"] == "latitude_longitude":
        projection = ccrs.PlateCarree()
    elif pr["grid_mapping_name"] == "lambert_azimuthal_equal_area":
        central_longitude = pr["longitude_of_projection_origin"]
        central_latitude = pr["latitude_of_projection_origin"]
        false_easting = pr["false_easting"]
        false_northing = pr["false_northing"]
        projection = ccrs.LambertAzimuthalEqualArea(
            central_longitude=central_longitude,
            central_latitude=central_latitude,
            false_easting=false_easting,
            false_northing=false_northing,
        )
    elif pr["grid_mapping_name"] == "lambert_conformal_conic":
        standard_parallel = pr["standard_parallel"]
        try:
            standard_parallel[1]
        except IndexError:
            standard_parallel = [standard_parallel, standard_parallel]
        projection = ccrs.LambertConformal(
            central_longitude=pr["longitude_of_central_meridian"],
            central_latitude=pr["latitude_of_projection_origin"],
            standard_parallels=standard_parallel,
        )
    else:
        raise NotImplementedError(f"Projection {pr['grid_mapping_name']} not known")

    return projection


def pmeshify(x, y):
    def oneD_expand(x):
        dx = np.diff(x.values)
        x_ = x[:-1] + dx / 2
        x_ = np.array([x[0] - dx[0] / 2, *x_, x[-1] + dx[-1] / 2])
        x_mid = (x_[:-1] + x_[1:]) / 2
        assert np.max(np.abs(x_mid - x)) < 1e-2 * np.min(
            np.abs(dx)
        ), "Midpoints do not match"
        return x_

    x = oneD_expand(x)
    y = oneD_expand(y)

    return x, y


def plot_var_unet(
    predictions,
    truths,
    title: str | None = None,
    extents: list[float] | None = None,
    map_projection: ccrs.Projection = None,
    minval: float = None,
    maxval: float = None,
):  
    # Use SNAP file as template for plotting
    snap_file_path = f'data/ringhals_20230101_00Z.nc'
    snap_data = xr.open_dataset(snap_file_path, engine="netcdf4")
    snap_data = snap_data.isel(x=slice(188, 524), y=slice(97, 433)) # Dimensions: 336x336
    model = snap_data

    # Get the projection of the SNAP data
    data_projection = get_projection(model)

    # Get random samples from test set
    indices = random.sample(range(0, len(predictions)), 3)
    
    # Create 3x2 figure
    fig, axes = plt.subplots(
        3, 2,
        subplot_kw={
            "projection": map_projection if map_projection is not None else data_projection
        },
        figsize=(12, 18)
    )
    
    # Flatten axes for easy iteration
    axes = axes.flatten()

    # Use a built-in colormap for a smooth gradient
    cmap = plt.get_cmap('viridis')
    
    norm = mcolors.Normalize(vmin=minval, vmax=maxval)

    for i, index in enumerate(indices):
        # Get prediction and truth for this sample
        unet_pred = predictions[index][0] - 20
        snap_truth = truths[index][0] - 20
        
        # Prediction plot
        ax = axes[i*2]

        # Rewrap the NumPy array into an xarray.DataArray with coordinates
        unet_pred_wrapped = xr.DataArray(
            unet_pred,
            dims=["x", "y"],
            coords={"x": snap_data["x"], "y": snap_data["y"]},
            name="restored_data")
        
        # Mask out values below minval
        mvar = np.ma.masked_where(unet_pred_wrapped.values <= minval, unet_pred_wrapped.values)
        x, y = pmeshify(unet_pred_wrapped["x"], unet_pred_wrapped["y"])
        
        # Plot the data
        mesh = ax.pcolormesh(x, y, mvar, cmap=cmap, norm=norm, transform=map_projection)
        iou_score = int_over_un(threshold=-19, pred=unet_pred, truth=snap_truth)
        ax.set_title(f'Test Sample {index} - Prediction. IoU: {iou_score:.2f}')
        
        # Truth plot
        ax = axes[i*2 + 1]

        # Rewrap the NumPy array into an xarray.DataArray with coordinates
        snap_truth_wrapped = xr.DataArray(
            snap_truth,
            dims=["x", "y"],
            coords={"x": snap_data["x"], "y": snap_data["y"]},
            name="restored_data")
        
        # Mask out values below minval
        mvar = np.ma.masked_where(snap_truth_wrapped.values <= minval, snap_truth_wrapped.values)
        
        # Plot the data
        mesh = ax.pcolormesh(x, y, mvar, cmap=cmap, norm=norm, transform=map_projection)
        ax.set_title(f'Test Sample {index} - Truth')
        
        # Add features to both subplots
        for j in range(2):
            ax = axes[i*2 + j]
            ax.add_feature(cfeature.COASTLINE)
            ax.add_feature(
                cfeature.LAKES, zorder=-10, edgecolor="none", facecolor=[0.0, 0.0, 0.8, 0.2]
            )
            ax.add_feature(
                cfeature.LAKES, zorder=2, edgecolor="k", linewidth=0.5, facecolor="none"
            )
            ax.add_feature(cfeature.RIVERS, edgecolor=[0, 0, 0.8, 0.2])
            ax.add_feature(
                cfeature.NaturalEarthFeature(
                    category="cultural",
                    name="admin_0_countries_deu",
                    scale="10m",
                    facecolor="none",
                ),
                edgecolor="black",
                zorder=14,
            )
            ax.gridlines(draw_labels=["left", "bottom"], x_inline=False, y_inline=False)
            
            if extents is not None:
                ax.set_extent(extents, crs=map_projection)

    fig.colorbar(mesh,  ax=axes[:], orientation='horizontal', fraction=0.05, pad=0.05, label=r'Accumulated Concentration ($\ln(\mathrm{g}/\mathrm{m}^2)$)')

    if title is not None:
        fig.suptitle(title, y=0.95)

    plt.savefig('figures/prediction_examples.png')
    
    plt.show()

    plt.close()

def int_over_un(threshold, pred, truth):

    pred_binary = np.where(pred > threshold, 1, 0)
    truth_binary = np.where(truth > threshold, 1, 0)
    
    intersection = np.logical_and(pred_binary == 1, truth_binary == 1)
    union = np.logical_or(pred_binary == 1, truth_binary == 1)

    if np.sum(union) == 0:
        return 1.0 if np.sum(intersection) == 0 else 0.0  # Handle edge case: both are all zeros
    
    iou = np.sum(intersection)/np.sum(union)

    return iou


def permutation_importance(model, X, y_true, metric='mse'):

    # Ensure tensors
    if isinstance(X, np.ndarray):
        X = torch.tensor(X, dtype=torch.float32)
    if isinstance(y_true, np.ndarray):
        y_true = torch.tensor(y_true, dtype=torch.float32)

    model.eval()
    device = next(model.parameters()).device
    X = X.to(device)
    y_true = y_true.to(device)

    N, C, H, W = X.shape
    base_errors = []

    # Baseline error with unshuffled input
    for i in range(N):
        xi = X[i].unsqueeze(0)  # (1, 8, H, W)
        yi = y_true[i].unsqueeze(0)  # (1, 1, H, W)
        with torch.no_grad():
            y_pred = model(xi)
            if metric == 'mse':
                error = F.mse_loss(y_pred, yi).item() 
            elif metric == 'iou':
                error = int_over_un(threshold=0.5, pred=y_pred, truth=yi).item()
            else:
                ValueError
        base_errors.append(error)

    base_error_mean = np.mean(base_errors)
    importances = []

    for var_idx in range(C):
        errors = []
        for i in tqdm(range(N), desc=f"Permuting variable {var_idx}"):
            xi = X[i].clone()  # shape: (8, H, W)
            yi = y_true[i].unsqueeze(0)

            # Shuffle the selected variable across the dataset
            shuffled_vals = X[:, var_idx].reshape(N, -1)
            perm = torch.randperm(N)
            shuffled = shuffled_vals[perm].reshape(N, H, W)
            xi[var_idx] = shuffled[i]  # Replace the variable with permuted values

            xi = xi.unsqueeze(0)  # (1, 8, H, W)

            with torch.no_grad():
                y_pred = model(xi)
                if metric == 'mse':
                    error = F.mse_loss(y_pred, yi).item() 
                elif metric == 'iou':
                    error = int_over_un(threshold=0.5, pred=y_pred, truth=yi).item()
                else:
                    ValueError
            errors.append(error)

        delta_error = -(np.mean(errors) - base_error_mean)
        importances.append(delta_error)

    return importances

def plot_importances(importances, metric='mse'):
    # Names for the 8 input variables
    variable_names = [
        'x-wind (T)',
        'x-wind (T+6h)',
        'y-wind (T)',
        'y-wind (T+6h)',
        'ABL height (T)',
        'ABL height (T+6h)',
        'Surf. Pressure (T)',
        'Surf. Pressure (T+6h)'
    ]

    # Plot
    plt.figure(figsize=(8, 6))
    plt.barh(variable_names, importances, color='steelblue')

    if metric=='mse':
        plt.xlabel("Increase in MSE (Permutation Importance)", fontsize=12)
    elif metric=='iou':
        plt.xlabel("Drop in IoU (Permutation Importance)", fontsize=12)

    plt.title("Importance of Meteorological Variables for U-Net Emulator", fontsize=14)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.gca().invert_yaxis()  
    plt.tight_layout()
    plt.savefig(f'figures/{metric}_variable_importance.png')
    plt.show()

def unet_predict(model, X_data, y_data):

    model.eval()

    num_samples = X_data.shape[0]
    # Preallocate NumPy arrays for predictions and truths
    pred_conc_array = np.empty((num_samples, *y_data.shape[1:]))  # Define output_shape based on model output
    true_conc_array = np.empty((num_samples, *y_data.shape[1:]))  

    for sample in range(num_samples):
        
        # Get truth
        truth_conc = y_data[sample]
        
        # Generate prediction from model
        input_stack = X_data[sample]
        
        # Convert to tensor (add batch dimension)
        input_tensor = torch.from_numpy(input_stack[np.newaxis, ...]).float()
        
        # Make prediction
        with torch.no_grad():
            output_tensor = model(input_tensor)
        
        # Get prediction and store in pre-allocated array
        pred_conc = output_tensor.numpy().squeeze()
        
        pred_conc_array[sample] = pred_conc
        true_conc_array[sample] = truth_conc

    return pred_conc_array, true_conc_array


def plot_correlation(pred_fields, truth_fields):
    # Adjust values by subtracting 20
    pred_fields -= 20
    truth_fields -= 20

    # Set a minimum value to filter out
    minval = -20

    # Create a mask to filter out very small/zero values
    mask = (pred_fields >= minval) & (truth_fields >= minval)
    pred_filtered = pred_fields[mask] 
    truth_filtered = truth_fields[mask]

    # Calculate Pearson correlation
    corr_coef, _ = pearsonr(pred_filtered.flatten(), truth_filtered.flatten())

    max_val = max(np.max(pred_filtered), np.max(truth_filtered))

    # Create the heatmap
    plt.figure(figsize=(8, 8))
    heatmap, xedges, yedges = np.histogram2d(
        pred_filtered.flatten(), 
        truth_filtered.flatten(), 
        bins=40,
        range=[[minval, max_val], [minval, max_val]]  # Explicit range
    )

    # Plot heatmap (with optional log scaling)
    plt.imshow(
        heatmap.T, 
        origin='lower', 
        cmap='viridis', 
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        aspect='auto',
        vmax=800
    )

    # Plot y = x line
    plt.plot([minval, max_val], [minval, max_val], 'r--', label='y=x')

    # Labels & title
    plt.xlabel('Predicted Values')
    plt.ylabel('True Values')
    plt.title(f'Heatmap of Predicted vs True Values\nCorrelation: {corr_coef:.2f}')
    plt.colorbar(label='Number of points')
    plt.legend()
    plt.grid()

    # Ensure axis limits include all data
    plt.xlim(minval, max_val)
    plt.ylim(minval, max_val)

    plt.savefig('figures/correlation_plot.png')
    plt.show()

    return corr_coef

if __name__ == "__main__":
    # Load state_dict 
    model_weights_path = f'models/unet_emulator_{DATA_MODE}.pth'
    model = unet.UNET()  # Recreate model architecture
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(model_weights_path, map_location=device))

    # load data
    print('Loading Data')
    X_data = np.load(f'data/test_set_{DATA_MODE}.npz')['pred_vars']
    y_test = np.load(f'data/test_set_{DATA_MODE}.npz')['target_vars']

    print('Calculating IoU Permutation Importances')
    importances_iou  = permutation_importance(model, X_data, y_test, metric='iou')
    print('Calculating MSE Permutation Importances')
    importances_mse  = permutation_importance(model, X_data, y_test, metric='mse')

    # Produce plots for permutation importance
    plot_importances(importances_iou, metric='iou')
    plot_importances(importances_mse, metric='mse')

    pred_conc_array, true_conc_array = unet_predict(model, X_data, y_test)

    # Produce plot for examples of emulator predictions
    plot_var_unet(pred_conc_array.copy(), true_conc_array.copy(), maxval=0, minval=-19)

    print('Calculating IoU Values for Testing Dataset')
    iou_vals = []
    for sample in range(pred_conc_array.shape[0]):
        prediction = pred_conc_array[sample, 0]
        truth = true_conc_array[sample, 0]
        iou = int_over_un(1, prediction, truth)
        iou_vals.append(iou.item())

    avg_iou = np.mean(iou_vals)
    print('Average IoU Score from test set: ' + str(avg_iou))

    corr = plot_correlation(pred_conc_array, true_conc_array)

    print('Pearson Correlation from test set: ' + str(corr))




