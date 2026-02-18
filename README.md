# Housing Price Predictor

This repository contains a Streamlit application to predict California housing prices using your saved model and scaler.

## Provided Results

- Mean Squared Error: 4974811836.391974
- R-squared: 0.6266814954463519
- Adjusted R-squared: 0.6258024526100224

## Usage

1. Place your `linear_model.pkl` and `scaler.pkl` in the project root (or upload them via the app UI).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:

```bash
streamlit run app.py
```

Open the URL printed by Streamlit in your browser, adjust input features and press *Predict* to see the estimated median house price.
