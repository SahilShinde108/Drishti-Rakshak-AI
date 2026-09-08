import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import numpy as np
import logging
import yaml
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_config(config_name: str = "pipeline_config.yaml") -> dict:
    config_path = Path(__file__).resolve().parents[2] / "configs" / config_name
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}

class DenoisingAutoencoder(nn.Module):
    def __init__(self, in_channels=1, latent_dim=64):
        super(DenoisingAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, latent_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(latent_dim),
            nn.ReLU(True)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class DAETrainer:
    def __init__(self, model: DenoisingAutoencoder):
        self.config = load_config()
        self.model = model.to(DEVICE)
        self.lr = self.config.get("training", {}).get("dae_lr", 1e-3)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def train(self, dataloader, epochs: int):
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in dataloader:
                inputs, targets = batch # Noisy inputs, clean targets
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                
            logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")

    def save_checkpoint(self, path: str):
        torch.save(self.model.state_dict(), path)

class DAEDenoiser:
    def __init__(self, weights_path: Optional[str] = None):
        self.config = load_config()
        in_channels = self.config.get("preprocessing", {}).get("channels", 1)
        latent_dim = self.config.get("preprocessing", {}).get("dae_latent_dim", 64)
        
        self.model = DenoisingAutoencoder(in_channels, latent_dim).to(DEVICE)
        self.is_demo_mode = True
        
        if weights_path and Path(weights_path).exists():
            self.model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
            self.model.eval()
            self.is_demo_mode = False
            logger.info("Loaded DAE weights. Running in production mode.")
        else:
            logger.warning("DAE weights not found. Running in DEMO/FALLBACK mode.")

    def denoise(self, image: np.ndarray) -> np.ndarray:
        if self.is_demo_mode:
            # Fallback: Bilateral + Non-Local Means
            logger.info("[DEMO] Applying traditional denoising filters instead of DAE.")
            if image.dtype == np.float32 or image.dtype == np.float64:
                img_8u = (image * 255).astype(np.uint8)
            else:
                img_8u = image
                
            bilateral = cv2.bilateralFilter(img_8u, 9, 75, 75)
            nlm = cv2.fastNlMeansDenoising(bilateral, h=10, searchWindowSize=21, templateWindowSize=7)
            
            if image.dtype == np.float32 or image.dtype == np.float64:
                return nlm.astype(np.float32) / 255.0
            return nlm
            
        # Production Mode
        self.model.eval()
        with torch.no_grad():
            tensor_img = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
            if len(image.shape) == 3: # color
                tensor_img = torch.from_numpy(image).float().permute(2,0,1).unsqueeze(0).to(DEVICE)
            
            output = self.model(tensor_img)
            
            out_img = output.squeeze().cpu().numpy()
            if len(image.shape) == 3:
                out_img = np.transpose(out_img, (1, 2, 0))
            return out_img
