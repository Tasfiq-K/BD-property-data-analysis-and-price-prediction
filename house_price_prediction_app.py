import numpy as np
import pandas as pd
import streamlit as st
from functions import ct, trained_model, prediction_score


# links 
# https://docs.streamlit.io/library/api-reference/layout


df = pd.read_csv("data/property_listing_data_in_Bangladesh_new_mod1.csv")
df.drop(['Address', 'Sector_or_Block'], axis=1, inplace=True)

def town_selection(city):
    
    return sorted(df.loc[df['City'] == city].loc[:, 'Town'].unique())

def region_selection(town):
    
    return sorted(df.loc[df['Town'] == town].loc[:, 'Region'].unique())

def property_area(type):
    
    min_area, max_area = min(df.loc[df['Type'] == type]['Area']), max(df.loc[df['Type'] == type]['Area'])

    return st.slider("Area (Sqft.)", min_value=min_area, max_value=max_area, step=25)

mean_enc_town = df.copy().groupby('Town')['Price'].mean()
mean_enc_region = df.copy().groupby('Region')['Price'].mean()


def main():


    st.title('Bangladesh Property Price Prediction Project')
    st.write("""
    ## Property Information
    """)

    
    # st.dataframe(df.describe())

    container1 = st.container()
    with container1:
        col1, col2, col3, col4 = st.columns(4, gap='large')
        with col1:
            select_city = st.selectbox("Select City", df.loc[:, 'City'].unique())
        with col2:
            select_town = st.selectbox("Select Town", town_selection(select_city))
            enc_town = mean_enc_town[select_town]
        with col3:
            select_region = st.selectbox("Select Region", region_selection(select_town))
            enc_region = mean_enc_region[select_region]
        with col4:
            select_type = st.selectbox("Property Type", df.loc[:, 'Type'].unique())

        area = property_area(select_type)

    container2 = st.container()
    with container2:
        col1, col2 = st.columns(2, gap='large')
        with col1:
            n_beds = st.slider("Number of Beds", 
                            min_value=min(df.loc[:, 'Beds']),
                            max_value=max(df.loc[:, 'Beds'] + 4),
                            step=1
                            )
        with col2:
            n_baths = st.slider("Number of Baths", 
                                min_value=min(df.loc[:, 'Baths']),
                                max_value=max(df.loc[:, 'Baths'] + 2),
                                step=1
                                )
    # submitted = st.form_submit_button("Predict")

    n_rooms = n_beds + n_baths
    # st.write(f"Total {n_rooms} Rooms with {n_beds} Beds and {n_baths} Baths")

    infer_data = pd.DataFrame(data=[[select_type, select_city, n_beds, n_baths, n_rooms, area, enc_town, enc_region]],
                            columns=['Type', 'City', 'Beds', 'Baths', 'TotalRooms', 'Area', 'Town_mean_enc', 'Region_mean_enc'])
    transformer = ct()
    transformed_data = pd.DataFrame(data=transformer.transform(infer_data), columns=transformer.get_feature_names_out())
    model = trained_model()
    
    pred_price = prediction_score(model, transformed_data)

    if st.button("Predict Price"):
        st.write(f"Price for {n_beds} Beds, {n_baths} Baths in {select_town}, {select_region}, {select_city} is:")
        st.success(f"{pred_price[0]} BDT")
    
    # st.dataframe(infer_data)
    # st.dataframe(transformed_data)


if __name__ == '__main__':
    main()