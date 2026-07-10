# Regional-Language Audio Deepfake Detector

An end-to-end deep learning framework designed to detect synthetic voice-cloning variations targeting telephonic and social media messaging channels. This project utilizes a custom Convolutional Neural Network (CNN) built from scratch in PyTorch to distinguish authentic regional human voices from highly sophisticated generative AI voice clones.

---

## Table of Contents
* [The Problem and The Solution](#the-problem-and-the-solution)
* [Project Architecture](#project-architecture)
* [Technology Stack](#technology-stack)
* [Dataset Strategy](#dataset-strategy)
* [Installation and Usage](#installation-and-usage)

---

## The Problem and The Solution

### The Problem
Generative AI voice-cloning systems have lowered the barrier for bad actors to orchestrate highly targeted identity fraud and financial scams. By exploiting short, compressed audio snippets (such as WhatsApp voice notes or phone line recordings), these tools can bypass traditional audio filters built for studio-grade inputs, easily tricking human listeners.

### The Solution
This system introduces a localized defensive barrier. It accepts real-world, highly compressed multi-format audio notes (.wav, .mp3, .m4a, .ogg), converts their acoustic signals into visual frequency representations, and processes them through an optimized spatial neural network trained to detect microscopic digital anomalies, phase mismatches, and generative artifacts.

---

## Project Architecture

The application pipeline is broken into three core operational layers:

1. **Acoustic Preprocessing Pipeline:** Raw multi-format audio tracks are decoded, padded, and normalized dynamically using Librosa. The temporal data is transformed into Mel-Spectrogram matrices (128x173) to serve as structural grid inputs for the model.
2. **Deep Learning Model (PyTorch):** A custom 3-layer Convolutional Neural Network (torch.nn) that maps visual frequency alterations across time, abstracting boundaries between natural vocal cords and AI-generated outputs.
3. **Frontend Presentation Web App:** An interactive web dashboard constructed via Streamlit that enables drag-and-drop file ingestion, active sound playing features, and instantaneous probabilistic inference output scores.

---

## Technology Stack

* **Core Language:** Python
* **Deep Learning Framework:** PyTorch (torch, torch.nn, torch.utils.data)
* **Audio Engineering and Features:** Librosa, SciPy
* **Data Pipelines:** NumPy, Scikit-learn
* **Frontend Web Application:** Streamlit

---

## Dataset Strategy
* **Authentic Class (dataset/real):** Curated localized speech notes, containing natural background room environments, standard microphonic hiss, and regional pronunciation textures.
* **Synthetic Class (dataset/fake):** Advanced, clean voice clones generated using modern structural algorithms.
* **Optimization:** Evaluated across a localized data split to achieve stable hyperparameter convergence over 15 epochs while actively suppressing validation overfitting.

---

## Installation and Usage

### 1. Prerequisites
Ensure you have the necessary dependencies installed:
```bash
pip install torch librosa streamlit numpy scipy scikit-learn
