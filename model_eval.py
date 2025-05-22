import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

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





