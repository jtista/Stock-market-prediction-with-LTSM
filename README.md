# Stock Market Prediction with LSTMs

Predicting American Airlines (AAL) stock prices with simple averaging baselines and a deep LSTM (Long Short-Term Memory) network, built in Python and TensorFlow.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](#) <!-- replace # with your Colab link -->

---

## Aim

The aim of this project is to find out how well a deep learning model can forecast a stock's price several days ahead, compared with simple statistical methods.

Stock prices are noisy, and simple methods can look very accurate when they only predict one day ahead. The real test is predicting **further into the future**, where the model has to use its own earlier predictions as inputs. This project:

1. Explores and visualises daily AAL price data
2. Builds two baseline models: a **standard moving average** and an **exponential moving average (EMA)**
3. Builds and trains a **3-layer LSTM** that forecasts the price **10 trading days ahead**
4. Compares the LSTM's predictions with the actual prices on test data the model has never seen

## Inspiration

This project is based on the DataCamp tutorial [LSTM in Python: Stock Market Predictions](https://www.datacamp.com/tutorial/lstm-python-stock-market). I adapted it to a different dataset and updated the code to work with current libraries (see [What I changed](#what-i-changed-from-the-tutorial)).

---

## Dataset

- **Source:** [Huge Stock Market Dataset](https://www.kaggle.com/datasets/borismarjanovic/price-volume-data-for-all-us-stocks-etfs) (Kaggle)
- **Stock:** American Airlines Group (AAL), file `aal.us.txt`
- **Period:** December 2013 to November 2017 (989 trading days)
- **Columns used:** Date, Open, High, Low, Close

The target is the **mid price**, the average of each day's High and Low: `(High + Low) / 2`.

> The dataset is not included in this repo. Download `aal.us.txt` from Kaggle and upload it to the notebook's working directory before running.

---

## Method

### 1. Data preparation
- **Chronological 80/20 split:** the first 791 days are used for training and the last 198 for testing. The data is split by time, not randomly, so the model never sees the future during training.
- **Normalisation:** prices are scaled to 0–1 with `MinMaxScaler`, fitted on 200-day windows of the **training data only** to avoid leaking test information.
- **Smoothing:** exponential moving average smoothing (γ = 0.1) is applied to the training data to reduce day-to-day noise.

### 2. Baseline models
| Model | How it predicts |
|---|---|
| Standard average | Mean of the previous 100 days |
| Exponential moving average | Weighted average where recent days count more (decay = 0.5) |

### 3. LSTM model
| Setting | Value |
|---|---|
| Layers | 3 LSTM layers (200, 200, 150 units) + linear output layer |
| Dropout | 0.2 |
| Sequence length (unrollings) | 50 days |
| Batch size | 50 |
| Epochs | 30 |
| Optimiser | Adam, learning rate 0.0001 with decay when test error stops improving |
| Forecast horizon | 10 days, with each prediction fed back in as the next input |

---

## Results

| Model | What's measured | MSE |
|---|---|---|
| Standard average | One day ahead, training data | 0.01975 |
| Exponential moving average | One day ahead, training data | 0.000236 |
| **LSTM** | **10 days ahead, unseen test data** | **0.00603** (best epoch: 12) |

**How to read these numbers:** the EMA's very low error looks impressive, but it only ever predicts **one day ahead**. Tomorrow's price is usually close to today's, so this is an easy task. Averaging methods can't produce multi-day forecasts, because they need the real price each day.

The LSTM solves a much harder problem. It forecasts **10 days ahead on data it has never seen**, using its own predictions as inputs, and still achieves a low error. This is the main takeaway from the project.

### Mid price over time
<img width="1789" height="890" alt="MidPrice" src="https://github.com/user-attachments/assets/81faaac3-ceab-40aa-b5c7-6e1e7bf1c50e" />


### LSTM test predictions
The top chart shows how the predictions improved across epochs (faint lines are early epochs, darker lines later ones). The bottom chart shows the best epoch. Each red segment is a separate 10-day forecast.
<img width="1476" height="1486" alt="LSTM pred" src="https://github.com/user-attachments/assets/e668fd7e-892a-41e3-bb17-7a36ac5911b8" />



---

## What I changed from the tutorial

- **Different dataset size:** the tutorial uses about 12,000 days of HP (HPQ) data. I used about 1,000 days of AAL, so I replaced all hard-coded values (the split point, scaling windows, test points and chart limits) with values **calculated from the data length**. The notebook now works with any stock file.
- **80/20 percentage split** instead of a fixed index (`mid_prices[:11000]`).
- **Batch size reduced from 500 to 50:** with a small dataset, a batch size of 500 gave only one training step per epoch, and the model didn't learn. At 50, it gets 15 steps per epoch and the test error dropped sharply.
- **TensorFlow 2 compatibility:** the tutorial was written for TensorFlow 1.6. Keras 3 removed `BasicLSTMCell` and `MultiRNNCell`, so I wrote custom LSTM, dropout and multi-layer cell classes that run in a `tf.compat.v1` graph.
- **Forecast settings:** 10-day forecasts starting every 10 days from the split point, so the test period is fully covered.
- **Best epoch selected automatically** with `np.argmin(test_mse_ot)` instead of being chosen by hand.
- **Date-based visualisations** with quarterly `YYYY-MM-DD` tick labels.

---

## Limitations and next steps

- **Small test set:** 198 days is one market period. Walk-forward validation (`TimeSeriesSplit`) would give a more reliable estimate across different market conditions.
- **Single feature:** the model only uses the mid price. Volume, technical indicators or market-wide data could improve it.
- **Longer history:** trying the full HPQ dataset (1970s onwards) would show how the model performs with much more training data.
- **Not financial advice:** this is a learning project. Predicting price levels in scaled units is not the same as making profitable trades.

---

## How to run

1. Open `Stock_Market_Predictions.ipynb` in Google Colab (or Jupyter).
2. Upload `aal.us.txt` from the Kaggle dataset.
3. Run all cells in order (**Runtime → Run all**).

**Libraries:** Python 3, NumPy, pandas, Matplotlib, scikit-learn, TensorFlow 2.x

---

## Author

**Joseph Bautista**: MSc Data Science and Analytics, University of Westminster
