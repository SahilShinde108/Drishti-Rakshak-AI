# Drishti-Rakshak AI

Drishti-Rakshak AI is a comprehensive AI-powered telemedicine framework for Diabetic Retinopathy (DR) and Diabetic Macular Edema (DME) screening in resource-constrained environments (like Indian PHCs).

## Architecture

```text
[PHC Edge] --> [Triage / Image Capture] --> [4G/3G Network] --> [District Hospital]
```

## Folder Structure
- `configs/`: YAML configuration files
- `data/`: Datasets
- `matlab_simulink/`: MATLAB scripts and simulation documentation
- `scripts/`: Training and evaluation scripts
- `src/`: Core Python modules (simulation, models, pipeline)
- `tests/`: Pytest test suites

## Environment Setup
- Python 3.10+
- `pip install -r requirements.txt`
- MATLAB R2023b+ (optional, for simulink)

## Dataset Preparation
Follow instructions to download:
- IDRiD
- Messidor-2
- APTOS 2019
Place them in `data/` as per the expected directory structure.

## Configuration
Modify `configs/pipeline_config.yaml` and `configs/models_config.yaml` for hyperparameters.

## Training Commands
```bash
python scripts/01_prepare_splits.py
python scripts/02_run_iqa_and_preproc.py
python scripts/03_train_models.py
python scripts/04_evaluate_system.py
```

## Demo Mode
Use the `--demo` flag for quick testing with synthetic data.

## Single Image Inference
```bash
python -m src.pipeline --image path/to/image.jpg
```

## Phase 2: Flask Application
A web portal for PHC technicians and ophthalmologists (coming soon).

## Reports and Integrations
- Auto-generated PDF reports with predictions and explanations.
- MATLAB / Simulink integration for queueing and IQA.
- SimPy telemedicine discrete event simulation.

## Testing
Run all tests:
```bash
python -m pytest tests/ -v
```

## Hardware Requirements
- CPU: Minimum (Intel Core i5 / AMD Ryzen 5)
- GPU: Recommended (NVIDIA with at least 8GB VRAM) for training

## Limitations
Predictions may be affected by extreme image quality degradation.

## Clinical Disclaimer
This software is for research and screening purposes only. It is not a replacement for a clinical diagnosis by a certified ophthalmologist.

## License
MIT License

## Contributors
Drishti-Rakshak Team
