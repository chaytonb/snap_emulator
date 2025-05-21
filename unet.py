import torch
from torch import nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import torch.nn.functional as F

class UNET(nn.Module):
    def __init__(self):
        super(UNET, self).__init__()
        
        # Input channels: 1 (concentration) + 4 (met vars) = 5
        self.input_channels = 5
        self.output_channels = 1

        # Pad input from 267x237 to 272x240
        self.pad = nn.ZeroPad2d(padding=(1, 2, 2, 3))  # (left, right, top, bottom)

        # Pooling function
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Encoder (Contracting Path)
        self.enc1 = self._double_conv(self.input_channels, 64) # 267 x 237
        self.enc2 = self._double_conv(64, 128) # 133 x 118
        self.enc3 = self._double_conv(128, 256) # 66 x 59
        self.enc4 = self._double_conv(256, 512)  # 33 x 29
        
        # Decoder (Expansive Path)
        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2) # 66 x 58
        self.dec4 = self._double_conv(512, 256) 

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2) # 132 x 116
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
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding='same'),
            nn.ReLU()
        )
        
    def forward(self, x):
        # Calls the various steps defined in init

        # pad input
        x = self.pad(x)

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

        # Crop back to [1, 267, 237]
        output = output[:, :, 2:-3, 1:-2]  # (top:2, bottom:3, left:1, right:2)
        
        return output
    
# Custom loss function
def physical_loss(pred, target):
    mse = F.mse_loss(pred, target)

    # Add mass conservation to loss term
    pred_mass = pred.sum()
    target_mass = target.sum()
    mass_err = F.mse_loss(pred_mass, target_mass)

    return mse #+ 0.3 * mass_err



def train(x_data_path, y_data_path, model_path):

    x_data = np.load(x_data_path)
    y_data = np.load(y_data_path)

    # Convert to PyTorch tensors
    X_tensor = torch.FloatTensor(x_data['arr_0'][0:900])
    y_tensor = torch.FloatTensor(y_data['arr_0'][:, np.newaxis, :, :][0:900]) # Add channel dimension

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X_tensor, y_tensor, test_size=0.2, random_state=42
    )

    train_loader = DataLoader(TensorDataset(X_train, y_train), shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), shuffle=True)

    # Initialize model, loss, optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNET().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

    # Training loop 
    num_epochs = 15
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = physical_loss(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                val_loss += physical_loss(outputs, targets).item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

    # Save the model
    torch.save(model.state_dict(), model_path)

if __name__ == "__main__":

    x_data_path = 'combined_preds.npz'
    
    y_data_path = 'combined_targets.npz'

    
    combined_file = train(x_data_path, y_data_path, 'models/unet_train_30days_30hr.pth')