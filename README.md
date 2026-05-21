<div align="center">
  <h1>🎭 Face Emotion Detection CNN</h1>
  <p>A deep learning project to detect facial emotions in real-time using OpenCV, TensorFlow, and Keras.</p>
</div>

---

## 📖 Overview

This project implements a Convolutional Neural Network (CNN) to classify facial expressions into one of seven categories:
**Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral.**

It provides a full end-to-end pipeline: from an exploratory Jupyter Notebook to a flexible training script and a **real-time webcam inference tool**.

![Sample Prediction](https://via.placeholder.com/600x300?text=Replace+with+your+WebCam+Screenshot)  
*(Upload a screenshot of your working webcam script here and replace the link!)*

## 🚀 Features

- **Automated Dataset Downloading:** Utilizes `kagglehub` to seamlessly fetch the FER2013 dataset.
- **Customizable Training CLI:** Train the model easily via command-line arguments.
- **Real-Time Detection:** A built-in OpenCV script that grabs frames from your webcam and detects emotions on the fly!

---

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/Face-Emotion-Detection.git
   cd Face-Emotion-Detection
   ```

2. **Install the dependencies:**
   Make sure you have Python 3.8+ installed. Run:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the Pre-Trained Model (Optional):**
   *(If you've hosted your `emotion_model.h5` in GitHub Releases)*
   Download the pre-trained `.h5` model from the [Releases page](https://github.com/your-username/Face-Emotion-Detection/releases) and place it in the root directory.

---

## 🖥️ Usage

### 1. Real-Time WebCam Inference

To see the model in action instantly, just run:
```bash
python webcam_inference.py
```
*Note: Make sure your webcam is enabled. Press **`q`** to exit the video stream.*

### 2. Static Image Inference

To predict the emotion on a specific image file:
```bash
python inference.py path/to/your/image.jpg
```

### 3. Training the Model from Scratch

You can train your own model using the `train.py` script. The script automatically downloads the FER2013 dataset and begins training. You can customize hyperparameters via the CLI:

```bash
python train.py --epochs 30 --batch_size 32 --model_name my_custom_model.h5
```

---

## 📊 Results and Evaluation

During training, the model achieves around **~62% validation accuracy** (which is standard for simple CNNs on the challenging FER2013 dataset). 

![Training Graph](https://via.placeholder.com/600x300?text=Replace+with+your+Accuracy/Loss+Graph)  
*(Save your matplotlib graphs from the notebook and link them here!)*

---

## 📂 Project Structure

```
├── .gitignore              # Ignored files (models, datasets)
├── README.md               # Project Documentation
├── requirements.txt        # Python dependencies
├── FACE_DETECTION.ipynb    # Original exploratory notebook
├── train.py                # Command-line training script
├── inference.py            # Static image prediction
└── webcam_inference.py     # Real-time OpenCV inference
```

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/your-username/Face-Emotion-Detection/issues).
