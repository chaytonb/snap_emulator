import torch
from torch import nn

class UNET(nn.Module):
    def __init__(self):
        super(UNET, self).__init__()
        
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
        
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2) 
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
        enc1_pool = self.pool(enc1) # 168x168
        
        enc2 = self.enc2(enc1_pool)
        enc2_pool = self.pool(enc2) # 84x84
        
        enc3 = self.enc3(enc2_pool)
        enc3_pool = self.pool(enc3) # 42x42

        enc4 = self.enc4(enc3_pool) 
    
        # Decoder with skip connections
        up4 = self.up4(enc4) # 84x84
        up4 = torch.cat((up4, enc3), dim=1)  # Skip connection
        dec4 = self.dec4(up4) 

        up3 = self.up3(dec4) # 168x168
        up3 = torch.cat((up3, enc2), dim=1)  # Skip connection
        dec3 = self.dec3(up3)
        
        up2 = self.up2(dec3) # 336x336
        up2 = torch.cat((up2, enc1), dim=1)  # Skip connection
        dec2 = self.dec2(up2)
        
        # Final output
        output = self.final(dec2) # 336x336
        
        return output 