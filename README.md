# DS_mini_project
This repository contains the source code of a mini-project done during the Introduction to Data Science course at the University of Helsinki. The application is a machine learning-based web-app that allows the user to create predictions of the availability of city bikes at a selected station on a specific day and time. The application uses HSL's (Helsinki Region Transport) open city bike data from 2016-2019.

## City bike availability prediction web-app

The application's UI is made using HTML, JavaScript, and Python with Flask. The application is run in a virtual environment (venv).

## Instructions for launching the application:


Clone this repository to your local machine. Go to the root of the directory where you cloned this repository. Install and activate the required virtual environment and application dependencies by running the commands below.
```
$ python3 -m venv venv

$ source venv/bin/activate

(venv) $ pip install -r requirements.txt
```

launch the application with the command
```
(venv) $ flask run
```
