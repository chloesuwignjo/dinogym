import streamlit as st
import numpy as np
import pandas as pd
import ssl

from PIL import Image
from pathlib import Path
from feature_extractor import FeatureExtractor 

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# streamlit layout
c1 = st.container()
c2 = st.container()
c3 = st.container()

col1, col2 = st.columns(2, gap='large')

# setup image classifier
fe = FeatureExtractor()
features = []
img_paths = []
for feature_path in Path("./feature").glob("*.npy"):
    features.append(np.load(feature_path))
    img_paths.append(Path("./img") / (feature_path.stem + ".jpg"))
features = np.array(features)

# read data
target_df = pd.read_csv("data/target_dino.csv")

def fun_fact(name):
    """
    Gets a fun fact about a given dinosaur.

    Args:
        name: a string representing dinosaur name.
    """
    dino_df = target_df[target_df['name'] == name]

    dino = dino_df.get('name').iloc[0].title()
    era = dino_df.get('period_name').iloc[0].title()
    year_start = int(dino_df.get('period_start').iloc[0])
    year_end = int(dino_df.get('period_end').iloc[0])
    country = dino_df.get('lived_in').iloc[0]
    species = dino_df.get('species').iloc[0].title()
    length = dino_df.get('length_meters').iloc[0]
    d_type = dino_df.get('type').iloc[0]
    named = dino_df.get('named').iloc[0]
    named_year = int(dino_df.get('named_year').iloc[0])
    if dino_df.get('diet_carnivorous').iloc[0] == True:
        diet = 'carnivore'
    elif dino_df.get('diet_omnivorous').iloc[0] == True:
        diet = 'omnivore'
    else:
        diet = 'herbivorous'
    
    return f"The {dino} existed during the {era} period from approximately {year_start} to {year_end} \
where it lived in current-day {country}. It is part of the species {species}, a(n) {d_type}. It is a \
{diet} dinosaur that can grow up to {length} meters! The {dino} was named by {named} in {named_year}."

# app header
with c1:
    st.title('Welcome to Prehistoric Pals!')
    uploaded_file = st.file_uploader("Upload an image to see which dinosaur your pet is most similar to!", \
                                     type=["jpg", "jpeg", "png"])

# check if image is uploaded
with c2:
    if uploaded_file is not None:
        # read image
        img = Image.open(uploaded_file)
        with col1:
            st.image(img, caption='Your super cute pet', width=280)
        
        # animal customizer
        with col2:
            animal_name = st.text_input("Pet name:")
            diet = st.radio("What's your pet's diet?", ['Herbivore', 'Carnivore', 'Omnivore'])
            origin = st.radio("Where does your pet come from?", ['Asia', 'Africa', 'Europe', 
                                                                'South America', 'North America',
                                                                'Oceania'])
            
        with col1:
            # set up button
            if st.button("Run!", type="primary"):
                # run search
                query = fe.extract(img)
                # L2 distances to features
                dists = np.linalg.norm(features - query, axis=1) 
                # get top results
                ids = np.argsort(dists)[:10]  
                scores = [(dists[id], img_paths[id]) for id in ids]

                # get search results
                results = []
                for i in range(10):
                    x=str((scores[i][1]))
                    y=(x.split('/')[1].split('_')[0])
                    results.append(y)

                # convert results to dictionary
                results_dict = {}
                for element in results:
                    if element not in results_dict.keys():
                        results_dict[element] = 1
                    else:
                        results_dict[element] += 1
                
                # get top matches
                sorted_dict = sorted(results_dict, key=lambda x: results_dict[x], reverse=True)
                top_matches = sorted_dict[:3]
                match1 = top_matches[0].lower()

                if not animal_name:
                    animal_name = 'Your pet'

                st.subheader(f'{animal_name} is most similar to a {match1}!')
                image_link = f"dino_pics/{match1}.png"
                st.image(image_link, caption = match1)
                st.write(fun_fact(match1))

                with col2:
                    new_dino = pd.read_csv('data/new_dino.csv')
                    X = new_dino.drop(columns=['avg_yr', 'length_meters'], axis=1)

                    y = new_dino['avg_yr'].drop(columns=['length_meters'])

                    # Split the data into training and testing sets
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

                    rf_regressor = RandomForestRegressor(n_estimators=100)

                    # Fit the model on the training data
                    rf_regressor.fit(X_train, y_train)

                    # Make predictions on the testing set
                    rf_predictions = rf_regressor.predict(X_test)

                    # Evaluate the model
                    rf_mse = mean_squared_error(y_test, rf_predictions)
                    rf_rmse = np.sqrt(rf_mse)
                    diet_supp = ['Herbivore', 'Carnivore', 'Omnivore']
                    continent_supp = ['Asia', 'Africa', 'Europe', 'South America', 'North America', 'Australia']
                    
                    # setting up the dataframe for the user input
                    con_af = 0
                    con_aq = 0
                    con_as = 0
                    con_eu = 0
                    con_na = 0
                    con_oc = 0
                    con_sa = 0

                    diet_car = 0
                    diet_herb = 0
                    diet_omnv = 0
                    
                    if diet == 'Herbivore':
                        diet_herb= 1
                    if diet == 'Carnivore':
                        diet_car = 1
                    if diet == 'Carnivore':
                        diet_omnv = 1

                    if origin == 'Asia':
                        con_as = 1
                    if origin == 'Africa':
                        con_af = 1
                    if origin == 'Europe':
                        con_eu = 1
                    if origin =='South America':
                        con_sa = 1
                    if origin == 'North America':
                        con_na = 1
                    if origin == 'Australia':
                        con_oc = 1                   

                    d = {'continent_AF': [con_af], 'continent_AQ': [con_aq], 'continent_AS': [con_as], 'continent_EU': [con_eu],
                        'continent_NA': [con_na], 'continent_OC': [con_oc], 'continent_SA': [con_sa], 'diet_carnivorous': [diet_car],
                        'diet_herbivorous': [diet_herb], 'diet_omnivorous': [diet_omnv], 'diet_unknown': [0]}
                    user_input = pd.DataFrame(data=d)

                    # checking to see what era the pet is in
                    pred_year = int(rf_regressor.predict(user_input))
                    if pred_year >= 201.5:
                        era = 'Triassic'
                    elif pred_year >= 145:
                        era = 'Jurassic'
                    else:
                        era = 'Mesozoic'
                    st.subheader(f'If {animal_name} is a dinosaur, {animal_name} would live approximately {pred_year} \
million years ago in the {era} era!')
                    
