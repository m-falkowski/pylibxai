# Pylibxai Audio Explainable AI Framework

## Description

The aim of this thesis is to design a tool that enables the execution, visualization, and comparison of Explainable Artificial Intelligence (XAI) methods for audio machine learning models. As part of this, the thesis describes the design of an audio XAI framework, which allows its user to integrate their models with the framework's API in the form of a library, which then performs explanations on the model and visualizes them in a web interface designed for the thesis. 

The thesis describes the developed framework architecture based on the Model-View-Controller (MVC) pattern, which aims to ensure the solution's modularity, enabling integration of new user models and alternative views to the default one. The project integrates selected XAI methods, including:

- **LIME** (Local Interpretable Model-agnostic Explanations)
- **Integrated Gradients**
- **LRP** (Layer-wise Relevance Propagation)

## Installation

### Prerequisites
- Python 3.9.20
- Conda package manager
- Git

### Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/m-falkowski/pylibxai
   ```

2. **Enter the cloned directory:**
   ```bash
   cd pylibxai/
   ```

3. **Initialize dependent repositories:**
   ```bash
   git submodule update --init --recursive
   ```

4. **Create and activate conda environment:**
   ```bash
   conda create -y --name pylibxai_env -c conda-forge python=3.9.20
   conda activate pylibxai_env
   ```

5. **Run the installation script:**
   ```bash
   chmod +x setup.sh
   sudo ./setup.sh
   ```

## Testing

The library provides a comprehensive testing suite that includes both unit tests and functional tests.

### Running Tests

Execute the test script from the command line:

```bash
./pylibxai_test.sh
```

This script performs:
- **Unit Tests**: Uses pytest to run unit tests for all framework components
- **Functional Tests**: Uses the framework's explanation runner script `pylibxai_explain.py` to test the entire explanation process end-to-end

### Test Coverage

The testing suite covers:
- Explainer components (LIME, Integrated Gradients, LRP)
- Interface validation and abstract method enforcement
- Context management and file operations
- View components and web interface functionality

## Usage

After installation, you can use the framework to explain audio model predictions:

```bash
python ./pylibxai/pylibxai_explain.py -w ./output_dir/ -m MODEL_NAME --explainer=lime,integrated-gradients --target=TARGET -i ./audio_file.wav
```

### Supported Models
- **CNN14** - [AudioSet Tagging CNN](https://github.com/qiuqiangkong/audioset_tagging_cnn)
- **HarmonicCNN (HCNN)** - [SOTA Music Tagging Models](https://github.com/minzwon/sota-music-tagging-models)
- **GtzanCNN** - Custom implementation for GTZAN genre classification

### Supported Explainers
- `lime` - Local Interpretable Model-agnostic Explanations
- `integrated-gradients` - Integrated Gradients method
- `lrp` - Layer-wise Relevance Propagation

## Architecture

The framework follows the Model-View-Presenter (MVP) architectural pattern:

- **Models**: Audio ML model adapters that implement explanation interfaces
- **Views**: Web-based and debug visualization components
- **Presenter**: Explainer components that orchestrate the explanation process

