# New York Subway Crowd Volume Predictor

Developed a Machine Learning Model that predicts the live subway crowd of the top 50 stations

![Dashboard Screenshot](image/dashboard.jpeg)

## Key Features
- **Interactive Map:** Leaflet.js map to select any subway station in NYC
- **Live Weather Integration:** Uses OpenWeatherMap API to adjust prediction based on the current temperature and precipitation
- **Deep Learning Model:** Uses a GRU (Gated Recurrent Unit) neural network to train the historical MTA data 
- **Real-time Dashboard:** Shows and updates the live predicted ridership counts 

## Tools Used
- **Frontend:** HTML, CSS, JavaScript (Leaflet.js)
- **Backend:** Python, Flask
- **Data Science:** Pandas, NumPy, Scikit-Learn
- **Machine Learning:** Random Forest Regressor, TensorFlow/Keras, GRU

## Files Explained
- **download.py:** Queried the NY State Socrata API to retrieve 2023 MTA data in monthly batches due to high volume of the dataset (https://data.ny.gov/Transportation/MTA-Subway-Hourly-Ridership-2020-2024/wujg-7c2s/about_data)
- **data_preprocessing.ipynb:** Extracted top 50 stations from the 2023 MTA data and appended the corresponding weather data from meteostats
- **modeling.ipynb:** Developed and compared models in Random Forest Regressor, Basic Neural Network, and Recurrent Neural Network (GRU) 
- **modeling_with_weather.ipynb:** Added weather columns into modeling
- **app.py:** Backend of the website that uses the pre-trained GRU model from modeling_with_weather.ipynb
- **index.html:** Frontend of the website
- **station_coordinates.json:** station's coordinates used for locating the stations in the website
- **pkl files:** Saved scalers from modeling to use it in app.py
- **gru_model.keras:** GRU model saved for the website
