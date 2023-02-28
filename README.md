
# Machine Learning App

This is a Streamlit app that builds a machine learning model on the given data and makes predictions.


## How to Run the App

To run this app, please follow these steps:

1. Clone this repository to your local machine.

2. Open the terminal and navigate to the project directory.

3. Run the following command to install the required packages:

```bash
  pip install -r requirements.txt
```
Run the following command to start the app:

```bash
streamlit run Home.py
```    

## How to Use the App


1. Upload your data in CSV format.
2. Select the target variable.

3. Click the Run modelling button.

4. Download the best model for that data
## About the model

The app uses pycaret to build a machine learning model on the input data. The model choose from a lot of ML models, including Linear Regression, Random Forest, and Support Vector Regression the best possible one for that data.
After that it lets you download the model for future predictions.
