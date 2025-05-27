# Face Age Prediction

A deep learning project that predicts a person's age from facial images using a custom convolutional neural network (CNN).  
This application leverages PyTorch and OpenCV for image processing and inference, trained on the UTKFace dataset.

## Key Features

- Custom CNN Architecture: Implements `FAPNet`, a tailored CNN designed for facial age estimation.
- Human-Level Accuracy: Achieves a Mean Absolute Error (MAE) of 5.08 (train) and 5.16 (validation) — matching human-level performance.
- Trained on UTKFace: A diverse and widely used dataset for facial age prediction.
- Image Preprocessing: Uses OpenCV for robust face detection and image normalization.
- Interactive Notebooks: Includes Jupyter notebooks for exploration, training, and evaluation.
- App Interface: Easy-to-use interface for real-time prediction from uploaded images.

---

## Installation

### Prerequisites

- Python 3.7+
- pip

### Steps

1. Clone the Repository:

```bash
git clone https://github.com/zakariaelaoufi/Face-Age-Prediction.git
cd Face-Age-Prediction
```

2. Create a Virtual Environment (optional):

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```

3. Install Dependencies:

```bash
pip install -r requirements.txt
```

## Usage
### Run the App
```bash
python app.py
```
Upload a facial image and receive the predicted age in real time.

### Run the Notebook
```bash
jupyter notebook
```
Explore the core logic in:
- `Face_Age_Prediction.ipynb`
- `sample.ipynb`

## Project Structure

```bash
Face-Age-Prediction/
├── .devcontainer/            # Dev container config
├── .idea/                    # IDE configs
├── data/                     # UTKFace dataset (not included by default)
├── models/                   # Trained model checkpoints
├── Face_Age_Prediction.ipynb # Main notebook
├── sample.ipynb              # Sample testing notebook
├── app.py                    # Streamlit or Flask app (if applicable)
├── fapnet.py                 # Custom CNN model
├── image_processing.py       # Preprocessing utilities
├── requirements.txt          # Python dependencies
├── runtime.txt               # Runtime configuration
├── README.md                 # Project documentation
└── tetststts.jpg             # Sample test image
```

## Model: FAPNet

`FAPNet` is a lightweight yet effective CNN designed for age regression.  
It processes grayscale facial inputs and outputs a single scalar — the predicted age.

Trained using MSE loss, with a fine-tuned architecture for optimal bias-variance balance.

### Performance

| Metric          | Value |
|-----------------|-------|
| Train MAE       | 5.08  |
| Validation MAE  | 5.16  |

These results are comparable to human-level age estimation accuracy.

---

## Dataset: UTKFace

The model is trained on the [UTKFace dataset](https://susanqq.github.io/UTKFace/), a large-scale face dataset labeled by age, gender, and ethnicity. It contains over 20,000 images covering a wide age range (0–116 years), with diversity in ethnicity and gender.

**Note:** The dataset is not included in this repository. Please download it separately and place it in the `data/` directory.

---

## Contributing

Contributions are welcome!  
If you find a bug, want to suggest an improvement, or add a new feature, feel free to fork this repository and submit a pull request.

1. Fork the repository
2. Create a new branch: `git checkout -b feature-name`
3. Commit your changes: `git commit -m 'Add new feature'`
4. Push to the branch: `git push origin feature-name`
5. Open a pull request

---

## License

Feel free to use and modify it for personal or commercial use.

---

## Acknowledgments

- UTKFace Dataset by Zhang et al.
- PyTorch and OpenCV
- Open-source ML research community

---

**Made by [Zakaria Elaoufi](https://github.com/zakariaelaoufi)**
