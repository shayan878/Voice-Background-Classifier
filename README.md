# Audio Classification Network

A PyTorch-based neural network pipeline for audio classification, featuring background noise isolation, automated data augmentation, and MFCC feature extraction.

## Features
* **Audio Separation:** Uses `librosa.decompose.nn_filter` to isolate foreground vocals from background noise.
* **Automated Data Augmentation:** Implements programmatic pitch shifting, time stretching, background noise injection, and track shifting.
* **Dynamic Silence Removal:** Eliminates silent spans (below 20dB) dynamically prior to extraction.
* **Custom PyTorch Architecture:** A lightweight Sequential feed-forward neural network for binary audio classification (Vocal vs. Background).

## Project Structure
* `src/audio_utils.py`: Waveform manipulation, augmentations, and MFCC padding.
* `src/data_loader.py`: Dataset building and DataLoader initialization.
* `src/model.py`: PyTorch Neural Network architecture.
* `src/train.py`: Training loops and metrics tracking.
* `src/evaluate.py`: Testing routines and Confusion Matrix generation.
* `main.py`: Pipeline execution script.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/audio-classification.git
cd audio-classification
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Update the `DATA_PATH` variable in `main.py` to point to your directory of `.wav` files. 
2. Execute the pipeline:
```bash
python main.py
```

The script will process your data, extract 100x13 dimension MFCC features, execute the train/validation loops, and display a final Confusion Matrix of the test set results.
