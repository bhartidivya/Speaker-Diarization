# Speaker Diarization with pyannote.audio

This project builds and evaluates a speaker diarization workflow for medical
consultation audio from the
[Primock57 dataset](https://github.com/babylonhealth/primock57). The goal is to
identify "who spoke when" in each recording, evaluate model output against
annotated transcripts, and explore domain adaptation through fine-tuning.

The repository is organized as a notebook-based machine learning project. It
covers exploratory data analysis, zero-shot diarization with a pretrained
pyannote model, conversion/loading of TextGrid annotations, Diarization Error
Rate (DER) evaluation, and an initial fine-tuning workflow.

## Project Highlights

- Audio exploration with waveform, silence trimming, STFT, and mel-spectrogram
  visualizations.
- Zero-shot speaker diarization using `pyannote/speaker-diarization-3.1`.
- Ground-truth parsing from `.TextGrid` transcript annotations into pyannote
  `Annotation` objects.
- DER calculation with `pyannote.metrics`.
- Fine-tuning setup for the pretrained `pyannote/segmentation` model using
  PyTorch Lightning callbacks, early stopping, checkpointing, and Adam
  optimization.

## Dataset

The project uses Primock57, a collection of mock primary-care consultations
recorded with clinicians and simulated patients. The notebooks assume the data
is available in Google Drive with this structure:

```text
/content/drive/MyDrive/Speech_assignment/
  audio/
    *.wav
  transcripts/
    *.TextGrid
```

The assignment data is not committed to this repository, so update the notebook
paths if your local or Colab directory layout is different.

## Repository Structure

```text
.
|-- EDA.ipynb
|-- zero_shot_pretrained_pyannote_model_evaluation.ipynb
|-- fine_tuning.ipynb
`-- README.md
```

### `EDA.ipynb`

Performs exploratory analysis on sample consultation audio. It loads `.wav`
files with `librosa`, plays sample audio, trims silence, and visualizes raw
waveforms, zoomed waveforms, spectrograms, and mel-spectrograms.

### `zero_shot_pretrained_pyannote_model_evaluation.ipynb`

Runs the pretrained pyannote diarization pipeline on the dataset, loads
TextGrid transcript tiers as reference annotations, and computes DER for each
file. The notebook reports an average zero-shot DER of **21.8011%** using the
pretrained `pyannote/speaker-diarization-3.1` pipeline.

### `fine_tuning.ipynb`

Sets up a fine-tuning experiment for `pyannote/segmentation`. The notebook
configures a segmentation task, uses binary cross-entropy loss, trains with
Adam at `1e-4`, monitors validation performance, saves the best checkpoint, and
uses early stopping over a maximum of 20 epochs.

## Setup

These notebooks were designed to run in Google Colab.

1. Clone this repository.
2. Upload or mount the Primock57 data in Google Drive.
3. Install notebook dependencies:

```python
!pip install pyannote.audio gdown pydub praat-parselmouth pyannote.metrics textgrid
```

4. Configure a Hugging Face access token for pyannote models:

```python
HF_TOKEN = "your_hugging_face_token"
```

5. Replace the placeholder paths in the notebooks with your dataset location.

## Evaluation

The evaluation workflow compares model diarization output against TextGrid
ground-truth annotations using Diarization Error Rate:

1. Run the pretrained diarization pipeline on each `.wav` file.
2. Load the corresponding `.TextGrid` annotation.
3. Select the relevant speaker tier, such as `Doctor`, `Patient`, or `Speaker`.
4. Convert labeled intervals into pyannote `Annotation` segments.
5. Compute per-file and average DER with `DiarizationErrorRate`.

Current recorded baseline:

| Approach | Framework | Dataset Split/Scope | Metric |
| --- | --- | --- | --- |
| Zero-shot pretrained diarization | pyannote.audio | Primock57 audio files evaluated in notebook | DER: 21.8011% |
