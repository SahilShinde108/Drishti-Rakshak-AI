import simpy
import random
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional
import yaml
from pathlib import Path
from .bandwidth_latency_model import NetworkModel

logger = logging.getLogger(__name__)

def load_config(config_name: str = "pipeline_config.yaml") -> dict:
    try:
        config_path = Path(__file__).resolve().parents[2] / "configs" / config_name
        if config_path.exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
    except Exception as e:
        logger.warning(f"Failed to load config {config_name}, using defaults: {e}")
    return {}

@dataclass
class Patient:
    id: str
    phc_origin: str
    dr_stage: int
    is_emergency: bool
    arrival_time: float
    image_size_kb: float

@dataclass
class SimulationResults:
    total_patients_generated: int
    patients_filtered_at_edge: int
    patients_referred_to_hospital: int
    mean_queue_wait_time_minutes: float
    max_queue_wait_time_minutes: float
    mean_transmission_time_seconds: float
    doctor_utilization_percent: float
    emergency_patients_count: int
    emergency_mean_wait_time: float
    annual_capacity_estimate: int
    throughput_per_day: float

class PHCClinic:
    def __init__(self, location_id: str, patient_arrival_rate: float, has_edge_triage: bool):
        self.location_id = location_id
        self.patient_arrival_rate = patient_arrival_rate
        self.has_edge_triage = has_edge_triage

class DistrictHospital:
    def __init__(self, env: simpy.Environment, num_doctors: int, service_time_minutes: float):
        self.env = env
        self.num_doctors = num_doctors
        self.service_time_minutes = service_time_minutes
        self.doctors = simpy.PriorityResource(env, capacity=num_doctors)
        self.wait_times = []
        self.emergency_wait_times = []
        self.busy_time = 0.0

class TelemedicineSimulation:
    def __init__(self, config: Optional[dict] = None):
        if config is None:
            config = load_config()
        self.config = config.get("simulation", {})
        self.num_phcs = self.config.get("num_phcs", 50)
        self.num_doctors = self.config.get("num_doctors", 5)
        self.has_edge_triage = self.config.get("has_edge_triage", True)
        self.service_time = self.config.get("doctor_service_time_minutes", 5.0)
        
        self.env = simpy.Environment()
        self.hospital = DistrictHospital(self.env, self.num_doctors, self.service_time)
        
        self.stats = {
            'generated': 0,
            'filtered': 0,
            'referred': 0,
            'emergencies': 0,
            'tx_times': []
        }

    def generate_dr_stage(self) -> int:
        r = random.random()
        if r < 0.60: return random.choice([0, 1])
        elif r < 0.85: return 2
        elif r < 0.95: return 3
        else: return 4

    def patient_generator(self, phc: PHCClinic, network: NetworkModel):
        patient_id = 0
        while True:
            # Arrivals based on poisson process (exponential inter-arrival)
            # rate is patients per day, so inter-arrival is in days (convert to minutes)
            inter_arrival = random.expovariate(phc.patient_arrival_rate / (24 * 60))
            yield self.env.timeout(inter_arrival)
            
            stage = self.generate_dr_stage()
            is_emergency = (stage >= 3)
            size_kb = random.uniform(500, 2500) # 0.5 to 2.5 MB images
            
            patient = Patient(
                id=f"{phc.location_id}_{patient_id}",
                phc_origin=phc.location_id,
                dr_stage=stage,
                is_emergency=is_emergency,
                arrival_time=self.env.now,
                image_size_kb=size_kb
            )
            patient_id += 1
            self.stats['generated'] += 1
            
            if is_emergency:
                self.stats['emergencies'] += 1

            if phc.has_edge_triage and stage <= 1:
                # 60% chance normal cases are filtered locally if triage enabled
                # We know stage <= 1 is ~60% of cases
                self.stats['filtered'] += 1
                continue
                
            self.stats['referred'] += 1
            self.env.process(self.image_transmission(patient, network))

    def image_transmission(self, patient: Patient, network: NetworkModel):
        tx_time_sec = network.compute_transmission_time(patient.image_size_kb)
        yield self.env.timeout(tx_time_sec / 60.0) # wait in minutes
        self.stats['tx_times'].append(tx_time_sec)
        self.env.process(self.doctor_review(patient))

    def doctor_review(self, patient: Patient):
        arrive_hospital_time = self.env.now
        # Priority: 0 for emergency, 1 for regular
        priority = 0 if patient.is_emergency else 1
        
        with self.hospital.doctors.request(priority=priority) as req:
            yield req
            wait_time = self.env.now - arrive_hospital_time
            self.hospital.wait_times.append(wait_time)
            if patient.is_emergency:
                self.hospital.emergency_wait_times.append(wait_time)
                
            yield self.env.timeout(self.hospital.service_time_minutes)
            self.hospital.busy_time += self.hospital.service_time_minutes

    def run_simulation(self, duration_days: int = 365, network_scenario: str = '4g') -> SimulationResults:
        network = NetworkModel.get_scenario(network_scenario)
        
        phcs = [PHCClinic(f"PHC_{i}", patient_arrival_rate=10, has_edge_triage=self.has_edge_triage) 
                for i in range(self.num_phcs)]
                
        for phc in phcs:
            self.env.process(self.patient_generator(phc, network))
            
        self.env.run(until=duration_days * 24 * 60) # in minutes
        
        mean_wait = sum(self.hospital.wait_times) / max(1, len(self.hospital.wait_times))
        max_wait = max(self.hospital.wait_times) if self.hospital.wait_times else 0
        mean_tx = sum(self.stats['tx_times']) / max(1, len(self.stats['tx_times']))
        
        total_time = duration_days * 24 * 60
        utilization = (self.hospital.busy_time / (total_time * self.num_doctors)) * 100
        
        em_wait = sum(self.hospital.emergency_wait_times) / max(1, len(self.hospital.emergency_wait_times))
        
        return SimulationResults(
            total_patients_generated=self.stats['generated'],
            patients_filtered_at_edge=self.stats['filtered'],
            patients_referred_to_hospital=self.stats['referred'],
            mean_queue_wait_time_minutes=mean_wait,
            max_queue_wait_time_minutes=max_wait,
            mean_transmission_time_seconds=mean_tx,
            doctor_utilization_percent=utilization,
            emergency_patients_count=self.stats['emergencies'],
            emergency_mean_wait_time=em_wait,
            annual_capacity_estimate=int(self.stats['referred'] * (365 / max(1, duration_days))),
            throughput_per_day=self.stats['referred'] / max(1, duration_days)
        )

    def run_all_scenarios(self, duration_days: int = 365) -> dict:
        results = {}
        for scenario in ['2g', '3g', '4g']:
            sim = TelemedicineSimulation(self.config)
            results[scenario] = sim.run_simulation(duration_days, scenario)
        return results

    def generate_report(self, results: dict, save_path: str) -> str:
        report = "Telemedicine Queue Simulation Report\n"
        report += "="*40 + "\n\n"
        for scenario, res in results.items():
            report += f"Scenario: {scenario.upper()}\n"
            report += f"- Total Patients Generated: {res.total_patients_generated}\n"
            report += f"- Filtered at Edge: {res.patients_filtered_at_edge}\n"
            report += f"- Referred to Hospital: {res.patients_referred_to_hospital}\n"
            report += f"- Mean Wait Time (mins): {res.mean_queue_wait_time_minutes:.2f}\n"
            report += f"- Max Wait Time (mins): {res.max_queue_wait_time_minutes:.2f}\n"
            report += f"- Mean TX Time (secs): {res.mean_transmission_time_seconds:.2f}\n"
            report += f"- Doctor Utilization: {res.doctor_utilization_percent:.2f}%\n"
            report += "-"*20 + "\n"
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
        return report

if __name__ == "__main__":
    config = load_config()
    sim = TelemedicineSimulation(config)
    results = sim.run_all_scenarios(duration_days=30)
    print(sim.generate_report(results, ""))
