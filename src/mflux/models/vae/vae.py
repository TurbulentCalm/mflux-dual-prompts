import logging
import mlx.core as mx
from mlx import nn

from mflux.models.vae.decoder.decoder import Decoder
from mflux.models.vae.encoder.encoder import Encoder

log = logging.getLogger(__name__)

class VAE(nn.Module):
    scaling_factor: int = 0.3611
    shift_factor: int = 0.1159

    def __init__(self):
        super().__init__()
        self.decoder = Decoder()
        self.encoder = Encoder()

    def decode(self, latents: mx.array) -> mx.array:
        try:
            log.info(f"Starting VAE decode with latents shape {latents.shape}")
            scaled_latents = (latents / self.scaling_factor) + self.shift_factor
            log.info(f"Scaled latents shape: {scaled_latents.shape}")
            decoded = self.decoder(scaled_latents)
            log.info(f"Decoded shape: {decoded.shape}")
            return decoded
        except Exception as e:
            log.error(f"Error in VAE decode: {e}")
            return None

    def encode(self, latents: mx.array) -> mx.array:
        latents = self.encoder(latents)
        mean, _ = mx.split(latents, 2, axis=1)
        return (mean - self.shift_factor) * self.scaling_factor
