from app import app
import functions
from flask import render_template, request, redirect

@app.route("/")
def index():
    stations_list = functions.stations_dict('Data/station_data.csv')
    return render_template("predict.html", stations=stations_list)