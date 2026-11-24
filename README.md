# 🌿 Plant Disease Recognition

A deep-learning project that identifies a plant's disease from a photo of its leaf. A Convolutional Neural Network (CNN) was trained on leaf images covering **14 crop species and 38 classes** (diseased and healthy), and is served through a simple **Streamlit** web interface.

## Why this project
Plant diseases reduce crop yield and farmer income. Spotting them early, from just a phone photo, lets farmers act sooner (treatment, isolation, expert advice) and reduces losses.

## What it does
- Upload a leaf image (JPG/PNG)
- The model predicts the plant and its disease (or "Healthy")
- Shows the confidence and the top 3 predictions

## Supported crops
Apple, Blueberry, Cherry, Corn (maize), Grape, Orange, Peach, Bell Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato.

## Model
| Item | Detail |
|---|---|
| Framework | TensorFlow / Keras (Sequential CNN) |
| Input | 128 x 128 RGB image |
| Architecture | 5 blocks of 2 x Conv2D (32, 64, 128, 256, 512 filters) + MaxPooling, Dropout(0.25), Flatten, Dense(1500, ReLU), Dropout(0.4), Dense(38, Softmax) |
| Parameters | ~7.8 million |
| Optimizer / Loss | Adam (lr = 0.0001) / Categorical cross-entropy |
| Training | 10 epochs, batch size 32 |

## Results
| Set | Accuracy |
|---|---|
| Training | ~98.1% |
| Validation | ~95.1% |

(See `Train_plant_disease.ipynb` for the full classification report and confusion matrix.)

## Project structure
```
app.py                    # Streamlit interface
trained_model.keras       # trained model (output of the training notebook)
requirements.txt          # Python dependencies
Train_plant_disease.ipynb # model training and evaluation
Test_Plant_Disease.ipynb  # single-image prediction test
README.md
```

## How to run
```bash
python -m venv venv
venv\Scripts\activate        # Windows  (Linux/Mac: source venv/bin/activate)
pip install -r requirements.txt
streamlit run app.py
```
Open http://localhost:8501, upload a leaf image and click **Predict**.

## Notes
- Class order matches training (alphabetical folder names) and is defined in `app.py`.
- Images are resized to 128x128 and not rescaled, exactly as in training.
- Use the same TensorFlow version as in training if the model fails to load.
- Works best on a clear, single-leaf photo; real-field photos with cluttered backgrounds may be less accurate than the lab-style training images.

## Future improvements
- Treatment and prevention suggestions for each disease
- Camera capture and mobile-friendly deployment
- Training on real-field images and with data augmentation
