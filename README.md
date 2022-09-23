# Meteor detector by allskycams.com

A Tensorflow Object Detection model to detect meteors in the AllSky7 integrated system by Dennis Chicaiza.

## Required libraries
#
You must to install some libraries to proceed. Use a virtual environment if you prefer. Use the **requirements.txt** file to install all the packages needed.

```bash
pip install -r requirements.txt
```

This repository is used to run a Tensorflow Object Detection model to detect meteors only. For legal aspects the model is not in here.

Follow the next steps to run the model detections using a Flask application.

## Running the detection web app
#
### 1. Load the model

When the packages are installed. Insert the model into the root of the project. This folder is called *AllSkyNet* and contains the **checkpoint**, the **saved_model** and the **pipeline.config**.

### 2. Load the images

Create a folder called **/static**  into the root directory.

```bash
mkdir static
```

Load the images you want to use to detect into this static folder.

### 3. Run the flask app

When everything is correctly set into the project, run the Flask app using the next command.

```bash
python3 detect_allskynet.py
```

The AllSkyNet UI will appear and you will can select an image at a time from your static dataset to detect meteors.
<br>
<br>

## Future works
#
– Implementation of this amazing feature to the [AllSkyCams system](https://github.com/mikehankey/amscams).

– Use this model detection as a filter to save the images collected from the AllSkyCams and not as a single image detector.

<br>

## Acknowledgment
#
- Mike Hankey [[ref](https://github.com/mikehankey)]
- Tensorflow Object Detection API [[ref](https://github.com/tensorflow/models)]