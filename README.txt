Step 1: Set up Python Virtual Environments
Make sure you have Python 3.8+ and pip installed.
Then run the following for each folder:

#for flask folder
cd flask
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

#For the Detection Module
cd detection
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

#For the Grouping Module
cd grouping
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

Step 2: Run the Flask Web Application
cd flask
source venv/bin/activate
python app.py
#this will start the server on http://127.0.0.1:5001


#Run the Detection API server
cd detection
source venv/bin/activate
python detect_api.py

#Run the grouping code
cd grouping
source venv/bin/activate
python group.py


Upload a sample image.
The pipeline will:
    Detect objects in the image.
    Group visually similar products.
    Show you:
        Detected items.
        Grouping information.
        Grouped image visualization.
        Raw JSON data for both detections and groups.
        
        
        
NOTE: I am a linux user and the system does not let me install pip packages as they are externally managed. So it forced me to use venv for every module i worked on. That is the reason for different environments and the copies of dependencies. PLease make sure there is atleat 10gb free space as the nvidia cuda packages are very large. 

The initial plan of using docker images failed.
