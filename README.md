# CrossFit Couch Assistant

CrossFit Coach Assistant (CCA) is a software tool that facilitates the coach's analysis of strength and gymnastic exercises and the trainee's correction by displaying the results on a User Interface.

To achieve these goals, CCA offers 3 different types of analysis using Machine Learning and computer vision techniques:

- Body analysis 
- Symmetry analysis 
- Bar analysis

## Body analysis

With this analysis, you can see the keypoints postition in pixels of the body. You have to upload a video.

## Simmetry analysis

With this analysis, you can see the asimmetry in pixels between the right keypoints and the left keypoints of the body. You have to upload a frontal plane video.

## Bar analysis

With this analysis, you can see the bar postition and the keypoints postition of the body in pixels. You have to upload a side plane video.

# Usage

## Python setup

Create a virtual env or use your native python with version 3.9.To check your python version execute

`python -v`

## Install packages

To install python packages go to the project root and execute

`pip install -r requirements.txt`

## Run the application

Once all packages are installed, you only need to run the app using

`streamlit run app.py`

