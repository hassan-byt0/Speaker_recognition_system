
**Zero-shot Speaker Recognition or Identification**

### Project Overview

This project aims to develop a speaker recognition system similar to a face recognition system. The system will identify speakers based on their voice recordings. The provided datasets are used for training the model, creating a reference database, and testing the model's performance.

### Datasets

* **Train Dataset**: Contains labeled voice recordings of various speakers. This dataset is used to train the model.
* **Database Dataset**: Contains a set of voice recordings that act as the reference database. The goal is to match speakers in the test dataset to these recordings.
* **Test Dataset**: Contains voice recordings of speakers that need to be identified. The model should predict the speaker by matching these recordings to the database dataset.

### Project Structure

```
speaker-recognition
├── Dataset
│   ├── train.csv
│   ├── database.csv
│   ├── test.csv
|   |── test
|   |── train
├── app
│   ├── __init__.py
│   ├── data_processing.py
│   ├── model.py
│   ├── inference.py
│   ├── main.py
|   ├── requirements.txt
├── Dockerfile
└──  README.md
```

### Files Description

* **data_processing.py**: Contains functions for loading and processing the datasets.
* **model.py**: Contains functions for training the model and extracting speaker embeddings.
* **inference.py**: Contains functions for matching speakers and evaluating the model.
* **main.py**: The main script to run the entire pipeline.
* **Dockerfile**: Dockerfile to containerize the application.
* **requirements.txt**: List of dependencies required to run the project.
* **README.md**: This documentation file.
<!--* **report.md**: Detailed report of the approach, methodology, challenges faced, and improvements made.-->

### Getting Started

#### Prerequisites

* Docker installed on your machine.
* Git installed on your machine.

#### Cloning the Repository

```sh
git clone https://github.com/hassan-byt0/Speaker_recognition_system
cd speaker-recognition
```

#### Download the requirements.txt

```sh
pip download -r <path_to your_requirements.txt>
```

#### Building the Docker Image

```sh
docker build -t speaker-recognition .
```

#### Running the Docker Container

```sh
docker run -it --rm speaker-recognition
```

### Detailed Instructions

#### Data Processing

1. **Load Data**: Load the train, database, and test datasets from CSV files.
2. **Extract Features**: Preprocess the audio files and extract features required for model training.

#### Model Training

1. **Train the Model**: Use the training dataset to train the speaker recognition model.
2. **Save the Model**: Save the trained model for inference.

#### Inference

1. **Extract Embeddings**: Extract speaker embeddings from the database and test datasets using the trained model.
2. **Match Speakers**: Compare test embeddings with database embeddings to identify speakers.
3. **Evaluate**: Evaluate the model's performance using accuracy and F1 score.

### Evaluation

The model will be evaluated based on the following criteria:

* **Accuracy**: Proportion of correctly identified speakers.
* **F1 Score**: Harmonic mean of precision and recall.
* **Current Accuracy and F1 Score:** Accuracy: 0.1699 (17%), F1 Score: 0.1756 (17.6%)
<!--* **Dockerising the packages management**: Dockerise the speaker recognition system by building docker image id.-->

### Challenges Faced

* Handling variations in audio quality and background noise.
* Ensuring the model generalizes well to unseen speakers.
* Optimizing the model for better accuracy and F1 score.

### Additional Features

* Implemented Wav2vec2 pre-trained model for feature extraction by comparing waveforms as embeddings.
<!--* Used advanced preprocessing methods to enhance feature extraction.-->
* Dockerized the speaker recognition system by building a Docker image.

### Additional Comments

* Utilized the torchaudio model pre-trained Wav2vec2 model, which showcases significant improvements in terms of accuracy and F1 score. We can implement different open-source audio recognition models from Hugging Face models or other platforms.
* Attempted hyperparameter tuning by introducing training arguments from torch methods, but due to time constraints, it was not fully implemented.
* Utilized the Lightning.ai cloud platform in a Visual Studio setting as it provides vast computational power and is easy to use.



### License


---
