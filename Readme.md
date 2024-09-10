```md
# Action Recognition Project

This project focuses on **Action Recognition** using deep learning techniques, including transfer learning, optical flow-based motion recognition, data augmentation, and cross-validation. The models are trained on a subset of the **Stanford 40 Actions Dataset** and the **HMDB51 Dataset**.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

The goal of this project is to classify actions from videos using deep learning techniques. The project involves training three models:
1. **Model 1**: Initially trained on the Stanford 40 dataset (for transfer learning) and later fine-tuned on the middle frames extracted from the videos of the HMDB51 dataset. Data augmentation was applied to the images for better generalization.
2. **Model 2**: Trained on the optical flow data of the HMDB51 videos, using convolutional layers with residual connections.
3. **Model 3**: A fusion model that combines Model 1 and Model 2 by concatenating their flattened layers. Cross-validation was used to evaluate the performance of Model 3.

We used **OpenCV** and the **DIS method** for extracting the optical flow from the videos. Additionally, we incorporated some code from the TensorFlow website for Model 2.
https://www.tensorflow.org/tutorials/video/video_classification

## Dataset

- **Stanford 40 Actions Dataset**: Used for transfer learning in Model 1. Only a part of this dataset was used.
- **HMDB51 Dataset**: Used for action recognition in both Model 1 (middle frames) and Model 2 (optical flow). Only a part of this dataset was used.

## Installation

To set up the project, please follow the steps below:

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/action-recognition.git
   cd action-recognition
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure that you have a compatible version of TensorFlow installed and access to a GPU if available.

## Usage

1. Download and prepare the datasets (Stanford 40 for transfer learning and HMDB51 for action recognition).
   
2. Train the models:
   - **Model 1**: First trained on the Stanford 40 dataset with data augmentation and then fine-tuned on middle frames extracted from the HMDB51 dataset.
   - **Model 2**: Trains on optical flow data extracted from HMDB51 videos using OpenCV's DIS method.
   - **Model 3**: Combines the flattened layers of Model 1 and Model 2 to form a fusion model. Cross-validation is applied to evaluate performance.

3. Evaluate the models on the test set and view metrics such as confusion matrices, accuracy, precision, and recall.


## Model Architecture

### Model 1:
- **Initial Training**: Performed on the Stanford 40 dataset, which contains more image samples.
- **Fine-Tuning**: After training on Stanford 40, the model is fine-tuned on the middle frames of videos from the HMDB51 dataset.
- **Data Augmentation**: Applied to the image data for generalization and improved robustness.
- Layers used:
  - **Conv2D**: For feature extraction.
  - **Batch Normalization**: For normalization.
  - **Attention Layers**: To enhance learning by focusing on important regions in the frames.

### Model 2:
- Trained on the optical flow data of the HMDB51 videos.
- **Optical Flow Extraction**: OpenCV's **DIS method** was used to extract optical flow from the videos.
- Architecture:
  - A sequence of convolutional layers that apply the operation first over spatial dimensions, followed by temporal dimensions.
  - **Residual Connections**: To enhance learning stability and mitigate vanishing gradients.
  - **Multi-Head Attention**: Used to capture temporal dependencies more effectively across frames.

### Model 3:
- A fusion model created by concatenating the flattened layers of Model 1 and Model 2 to capture both spatial and motion information for improved action recognition.
- **Cross-Validation**: Cross-validation was applied to ensure robust evaluation of the model's performance across different subsets of data.

## Results

After training, each model is evaluated on the HMDB51 test set. The metrics used include accuracy, precision, recall, and confusion matrices.

- **Model 1**: Focuses on spatial features from images.
- **Model 2**: Captures motion through optical flow.
- **Model 3**: Combines both spatial and motion features, showing improved performance over the individual models. Cross-validation ensures consistent and reliable evaluation.

## Contributing

Feel free to open issues or submit pull requests for improvements.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
```
