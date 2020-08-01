#Disaster Response Pipeline

This repository contains files for the Disaster Pipeline project for Udacity.
There are three folders:

1. app - contains the API/HTML for the web app
2. data - contains the data sets used and the python file for the ETL pipeline
3. models - contains the file for the model pipeline

### To run the project:
You will need to take the following steps:
1. Run the following commands in the root directory to set up the database and the model:
    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
2. Run the following command in the app's directory to run the web app.
    - `python run.py`
3. Go to http://0.0.0.0:3001/ in a browser