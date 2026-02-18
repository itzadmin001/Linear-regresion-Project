# Housing Price Predictor

This repository contains a Streamlit application to predict California housing prices using your saved model and scaler.

## Provided Results

- Mean Squared Error: 3.843007e+09
- R-squared: 0.711614

## Usage

1. Place your `knn_model.pkl` and `scaler.pkl` in the project root (or upload them via the app UI).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:

```bash
streamlit run app.py
```

Open the URL printed by Streamlit in your browser, adjust input features and press *Predict* to see the estimated median house price.
