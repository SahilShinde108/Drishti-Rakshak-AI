import random
from typing import Dict
import yaml
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class NetworkModel:
    def __init__(self, bandwidth_kbps: float, latency_ms: float, packet_loss_rate: float):
        self.bandwidth_kbps = bandwidth_kbps
        self.latency_ms = latency_ms
        self.packet_loss_rate = packet_loss_rate

    def compute_transmission_time(self, image_size_kb: float) -> float:
        """
        Computes time to transmit an image.
        Time = (image_size_kb * 8) / bandwidth_kbps + latency_ms/1000
        Includes retransmissions based on packet loss.
        """
        base_time = (image_size_kb * 8) / self.bandwidth_kbps + (self.latency_ms / 1000.0)
        
        # Jitter: random variation +/- 20%
        jitter = base_time * random.uniform(-0.20, 0.20)
        total_time = base_time + jitter
        
        # Simulate packet loss retransmissions (simplified)
        attempts = 1
        while random.random() < self.packet_loss_rate:
            attempts += 1
            # Add base time again for retransmission
            total_time += base_time + (self.latency_ms / 1000.0)
            if attempts > 5:
                break # Cap attempts
                
        return max(0.0, total_time)

    def simulate_connection(self, num_images: int, mean_image_size_kb: float = 1500) -> Dict[str, float]:
        times = []
        failures = 0
        
        for _ in range(num_images):
            # Vary image size slightly
            size = random.normalvariate(mean_image_size_kb, mean_image_size_kb * 0.1)
            size = max(500, min(size, 3000))
            
            try:
                t = self.compute_transmission_time(size)
                times.append(t)
            except Exception:
                failures += 1
                
        return {
            "total_time_seconds": sum(times),
            "successful_transmissions": len(times),
            "failed_transmissions": failures,
            "mean_transmission_time": sum(times) / max(1, len(times)),
            "max_transmission_time": max(times) if times else 0.0,
            "min_transmission_time": min(times) if times else 0.0
        }

    @classmethod
    def get_scenario(cls, name: str) -> 'NetworkModel':
        scenarios = {
            '2g': {'bandwidth': 64, 'latency': 800, 'loss': 0.15},
            '3g': {'bandwidth': 2000, 'latency': 200, 'loss': 0.05},
            '4g': {'bandwidth': 20000, 'latency': 50, 'loss': 0.01}
        }
        s = scenarios.get(name.lower(), scenarios['3g'])
        return cls(s['bandwidth'], s['latency'], s['loss'])

    def compute_daily_capacity(self, available_hours: float = 8, mean_image_size_kb: float = 1500) -> int:
        available_seconds = available_hours * 3600
        mean_time = self.compute_transmission_time(mean_image_size_kb)
        if mean_time <= 0:
            return 0
        return int(available_seconds / mean_time)

if __name__ == "__main__":
    nm = NetworkModel.get_scenario('3g')
    res = nm.simulate_connection(100)
    print(res)
