<h1 align=center> BD property data analysis & price prediction </h1>
___

Analyzing the biggest property data of Bangladesh and predicting the price using ML models.<br>

**Check out** the Price prediction [app](https://tasfiq-k-bd-property-data-ana-house-price-prediction-app-woc4pb.streamlit.app)

## Project's Purpose
The purpose of this project is to analyze the property condition of Bangladesh and build an ML model to predict the price.

## Data
The dataset that was used in this project is publicly available in [kaggle](www.kaggle.com). Here's the direct [link](https://www.kaggle.com/datasets/ijajdatanerd/property-listing-data-in-bangladesh) to the dataset.

**Note**: The two csv files that are inside the `../data..` directory are preprocessed. 

## Work & Findings
Workflow contains **data analysis**, **feature engineering**, **preprocessing** and **training** ML models. <br>

Used *pandas* to wrangle and processing the data for analysis and *matplotlib* and *seaborn* for visualization (I kinda like seaborn)*Scikit-learn* is used to to do further preprocessing and training different ML models. <br> 

The dataset contains 7k+ property information of two big cities of Bangladesh, Dhaka and Chattogram. There are three types of properties, Apartment, Building and Duplex having most of being the apartment type (expected) and making the dataset an imbalance one. <br>

One of the key findings is that the **Area** feature which describes the square feet area of the property is the most correlated feature with the target (**Price**) variable. Check out the *bd_house_price_prediction.ipynb* file for the analysis part.

Tried a few regression model to train on the data. Used LinearRegression, DecisionTreeRegressor, RandomForestRegressor, Lasso, Elastic net etc. Also Used Xtreme gradient Boosing (xgb) algorithm, Light Gradient Boosting (lgb) algorithm. The RandomForestRegressor, xgb, lgb performed well while training. Also, I've also tried hyperparameter tuning on those models which I thought performed well. check out the *model_training.ipynb* file for the training part.

## Web App
I've also made an web app using [streamlit](www.streamlit.io) and deployed in streamlit. Do check out the [app](https://tasfiq-k-bd-property-data-ana-house-price-prediction-app-woc4pb.streamlit.app)

## End Note
The model is not well optimized. It is a work in progress. So, please do comment your feedback or suggestion. I'm also up for collaboration, so don't hesitate to make pull request.

