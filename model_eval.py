import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
import xarray as xr

class UNET(nn.Module):
    def __init__(self):
        super(UNET, self).__init__()
        
        # Input channels: 8, 4 met vars at T, 4 at T+6
        self.input_channels = 8
        self.output_channels = 1

        # Pooling function
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Encoder (Contracting Path)
        self.enc1 = self._double_conv(self.input_channels, 64) 
        self.enc2 = self._double_conv(64, 128) 
        self.enc3 = self._double_conv(128, 256) 
        self.enc4 = self._double_conv(256, 512)  
        
        # Decoder (Expansive Path)
        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2) # 
        self.dec4 = self._double_conv(512, 256) 

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2) # 
        self.dec3 = self._double_conv(256, 128)  # 128*2 because of skip connection
        
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2) # 264 x 232
        self.dec2 = self._double_conv(128, 64)  # 64*2 because of skip connection
        
        # Final layer
        self.final = nn.Sequential(
            nn.Conv2d(64, self.output_channels, kernel_size=1),
        )
        
    def _double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding='same'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding='same'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        
    def forward(self, x):
        # Calls the various steps defined in init

        # Encoder
        enc1 = self.enc1(x)
        enc1_pool = self.pool(enc1)
        
        enc2 = self.enc2(enc1_pool)
        enc2_pool = self.pool(enc2)
        
        enc3 = self.enc3(enc2_pool)
        enc3_pool = self.pool(enc3)

        enc4 = self.enc4(enc3_pool) # Don't apply pool to final encoding step
    
        # Decoder with skip connections
        up4 = self.up4(enc4) 
        up4 = torch.cat((up4, enc3), dim=1)  # Skip connection
        dec4 = self.dec4(up4) 

        up3 = self.up3(dec4)
        up3 = torch.cat((up3, enc2), dim=1)  # Skip connection
        dec3 = self.dec3(up3)
        
        up2 = self.up2(dec3)
        up2 = torch.cat((up2, enc1), dim=1)  # Skip connection
        dec2 = self.dec2(up2)
        
        # Final output
        output = self.final(dec2)
        
        return output
    
def int_over_un(pred_fields, truth_fields):

    threshold = 18

    num_samples = pred_fields.shape[0]

    ious = []

    for sample in range(num_samples):
        # Set values above the threshold to 0 and the rest to 1
        pred_binary = np.where(pred_fields[sample] > threshold, 0, 1)
        truth_binary = np.where(truth_fields[sample] > threshold, 0, 1)
        
        # Calculate overlap: Intersection of both binary arrays
        intersection = np.logical_and(pred_binary == 1, truth_binary == 1)
        union = np.logical_or(pred_binary == 1, truth_binary == 1)
        
        # Divide by areafraction representation
        iou = intersection/union

        ious.append(iou)

    return ious


if __name__ == "__main__":
    model_weights_path = 'models/unet_accumulated_t6_minus40.pth'

    # Load state_dict 
    model = UNET()  # Recreate model architecture
    model.load_state_dict(torch.load(model_weights_path, map_location=torch.device('cpu')))
    model.eval()  # set evaluation mode

    # Load data
    pred_data = np.load('scaled_combined_t6_data_accumulated.npz')['pred_vars'][-40:, 1:, :, :] # Last 40 set aside for testing, remove initial conc var
    targ_data = np.load('scaled_combined_t6_data_accumulated.npz')['target_vars'][-40:] # Last 40 set aside for testing

    num_samples = pred_data.shape[0]
    # Preallocate NumPy arrays for predictions and truths
    pred_conc_array = np.empty((num_samples, *targ_data.shape[1:]))  # Define output_shape based on your model's output
    true_conc_array = np.empty((num_samples, *targ_data.shape[1:]))  # Assume targ_data has shape (num_samples, height, width)

    for sample in range(num_samples):
        
        # Get truth
        truth_conc = targ_data[sample]
        
        # Generate prediction from model
        input_stack = pred_data[sample]
        
        # Convert to tensor (add batch dimension)
        input_tensor = torch.from_numpy(input_stack[np.newaxis, ...]).float()
        
        # Make prediction
        with torch.no_grad():
            output_tensor = model(input_tensor)
        
        # Get prediction and store in pre-allocated array
        pred_conc = output_tensor.numpy().squeeze()
        
        pred_conc_array[sample] = pred_conc
        true_conc_array[sample] = truth_conc

    spatial_acc = fomis(pred_conc_array, true_conc_array)
    print(spatial_acc)

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
    unit: str | None = None,
    map_projection: ccrs.Projection = None,
    figname: str | None = None,
    show_figure: bool = False,
    minval: float = None,
    maxval: float = None,
    num_levels: int = 10,
    scale: int = 'linear',
    var: str = 'PMCH_acc_concentration',
    ax = None
):  
    # Use SNAP file as template for plotting
    snap_file_path = f'output_files/ringhals_20230101_00Z.nc'
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
    
    if scale == 'log':
        norm = mcolors.LogNorm(vmin=minval, vmax=maxval)
    else:
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
        mesh = ax.pcolormesh(x, y, mvar[0], cmap=cmap, norm=norm, transform=map_projection)
        ax.set_title(f'Sample {index} - Prediction')
        
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
        mesh = ax.pcolormesh(x, y, mvar[0], cmap=cmap, norm=norm, transform=map_projection)
        ax.set_title(f'Sample {index} - Truth')
        
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

    # Add colorbar
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    fig.colorbar(mesh, cax=cbar_ax)
    cbar_ax.set_title(var, fontsize=8)

    # Add overall title if provided
    if title is not None:
        fig.suptitle(title, y=0.95)

    # Save or show figure
    if figname is not None:
        plt.savefig(figname, dpi=600, bbox_inches='tight')
    
    if show_figure:
        plt.show()

    plt.close()



