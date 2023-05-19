import numpy as np
import joblib

def mean_encode(df, columns, mean_with):

    df_mean_enc = df.copy()

    for col in columns:

        mean_enc = df_mean_enc.groupby(col)[mean_with].mean()
        # mean_enc_region = prop_data_mean_enc.groupby('Region')['Price'].mean()

        df_mean_enc.loc[:, col + "_mean_enc"] = df_mean_enc[col].map(mean_enc)
        # prop_data_mean_enc.loc[:, "Region_mean_enc"] = prop_data_mean_enc["Region"].map(mean_enc_region)
    return df_mean_enc

def ct():
    return joblib.load('transformer_obj.joblib')


def trained_model():
    return joblib.load('gridsearch_rf_0_1_0.joblib')

def prediction_score(model, input_data):
    prediction = np.expm1(model.predict(input_data))
    return np.round(prediction)


# prop_data = pd.read_csv('property_listing_data_in_Bangladesh_new_mod1.csv')
# prop_data.drop(['Address', 'Sector_or_Block'], axis=1, inplace=True)

# mean_enc_df = prop_data.copy()
