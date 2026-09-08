import streamlit as st
import pandas as pd
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.simulation.discrete_event_queue import TelemedicineSimulation, load_config
from ui.components import render_disclaimer, render_empty_state
from ui.constants import DISCLAIMER_TEXT

st.title("District Telemedicine Network Simulation")

col1, col2 = st.columns([1, 2])

with col1:
    with st.container(border=True):
        st.subheader("Network parameters", icon=":material/settings:")
        num_phcs = st.slider("Primary health centres", 10, 100, 50, key="sim_phcs")
        num_doctors = st.slider("District hospital doctors", 1, 20, 5, key="sim_doctors")
        network_speed = st.segmented_control("Network bandwidth", ["2G", "3G", "4G"], default="4G", key="sim_network")
        edge_triage = st.toggle("Edge triage (filter normal scans locally)", value=True, key="sim_edge")
        duration_days = st.slider("Simulation duration (days)", 1, 30, 7, key="sim_duration")
        
        run_sim = st.button("Run simulation", type="primary", icon=":material/play_arrow:")

        if run_sim:
            with st.status("Running SimPy simulation..."):
                config = load_config()
                if "simulation" not in config:
                    config["simulation"] = {}
                config["simulation"]["num_phcs"] = num_phcs
                config["simulation"]["num_doctors"] = num_doctors
                config["simulation"]["has_edge_triage"] = edge_triage
                
                sim = TelemedicineSimulation(config)
                net_scenario = network_speed.lower() if network_speed else "4g"
                res = sim.run_simulation(duration_days=duration_days, network_scenario=net_scenario)
                st.session_state["sim_res"] = res

with col2:
    if "sim_res" not in st.session_state:
        render_empty_state(
            icon=":material/analytics:",
            title="No simulation results",
            description="Configure parameters and run the simulation to see network performance."
        )
    else:
        res = st.session_state["sim_res"]
        
        with st.container(horizontal=True):
            st.metric("Total patients generated", res.total_patients_generated, border=True)
            st.metric("Filtered at edge", res.patients_filtered_at_edge, border=True)
            st.metric("Referred to hospital", res.patients_referred_to_hospital, border=True)
            
        with st.container(horizontal=True):
            st.metric("Mean wait time (mins)", f"{res.mean_queue_wait_time_minutes:.1f}", border=True)
            st.metric("Doctor utilization", f"{res.doctor_utilization_percent:.1f}%", border=True)
            st.metric("Mean TX time (secs)", f"{res.mean_transmission_time_seconds:.1f}", border=True)
            
        with st.container(horizontal=True):
            st.metric("Emergency patients", res.emergency_patients_count, border=True)
            st.metric("Emergency wait (mins)", f"{res.emergency_mean_wait_time:.1f}", border=True)
            st.metric("Max wait (mins)", f"{res.max_queue_wait_time_minutes:.1f}", border=True)
            st.metric("Throughput / day", f"{res.throughput_per_day:.1f}", border=True)
            
        chart_df = pd.DataFrame({
            "Category": ["Generated", "Filtered (edge)", "Referred (hospital)", "Emergency"],
            "Patients": [res.total_patients_generated, res.patients_filtered_at_edge, res.patients_referred_to_hospital, res.emergency_patients_count]
        }).set_index("Category")
        st.bar_chart(chart_df)

with st.container(border=True):
    st.caption("Screening pathway")
    st.markdown("**PHC** :material/arrow_forward: **AI edge triage** :material/arrow_forward: **District hospital** :material/arrow_forward: **Ophthalmologist**")

render_disclaimer()
