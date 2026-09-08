# Drishti-Rakshak AI: MATLAB & Simulink Integration

This directory contains MATLAB scripts and documentation for building the Simulink/SimEvents model.

## Prerequisites
- MATLAB R2023b or later recommended
- Toolboxes: Image Processing, Computer Vision, Deep Learning, Medical Imaging, Statistics and Machine Learning

## Available Scripts (`scripts/`)
1. `IQA_Assessment.m`: Image Quality Assessment metrics (BRISQUE, NIQE, etc.)
2. `Preprocessing_7Step.m`: 7-step enhancement pipeline
3. `Retinal_Segmentation.m`: Vessel, disc, and cup segmentation
4. `DL_GradCAM_Alignment.m`: Grad-CAM overlay and lesion mask IoU
5. `Run_District_Simulation.m`: Pure MATLAB fallback for the district telemedicine queue simulation

## How to Construct the Simulink SimEvents Model
Note: binary `.slx` files are not provided. You can construct the `telemedicine_district_queue.slx` model using these steps:

1. Open MATLAB and start Simulink.
2. Create a new SimEvents Model.
3. **Entity Generator**: Represents PHC patient arrivals.
   - Set Time Action to random exponential to model Poisson arrivals.
   - Generate attributes for `dr_stage` and `is_emergency`.
4. **Stateflow Chart (Triage)**:
   - Route entities. If `dr_stage <= 1`, route to "Filtered Terminater".
   - Else, route to Network Transmission.
5. **Entity Server (Network Transmission)**:
   - Represents the 4G/3G network. Set service time to represent latency + bandwidth transmission delay.
6. **Entity Queue (Hospital Waiting Room)**:
   - Sort entities by priority (`is_emergency`).
7. **Entity Server (Doctors)**:
   - Set capacity to `num_doctors`.
   - Set service time (e.g., normal distribution around 5 minutes).
8. **Entity Terminator**: End of the line. Add scopes and statistic blocks to measure queue length and wait times.

## Expected Outputs
The scripts and the Simulink model will output capacity estimations, utilization percentages, and wait times to validate the scalability of the telemedicine network.
