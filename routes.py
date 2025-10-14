from app import app
from datetime import  datetime
import functions
import functions_mysha
from flask import render_template, request, redirect

@app.route("/")
def index():
    stations_list = functions.stations_dict('Data/station_data.csv')
    return render_template("predict.html", stations=stations_list)


@app.route("/create_prediction", methods=["GET", "POST"])
def create_prediction():
    if request.method == "GET":
        return render_template("predict.html")
    if request.method == "POST":
        station_name = request.form.get("station_name")
        station_id = int(request.form.get("station_id"))
        station_capacity = int(request.form.get("station_capacity"))
        date_time_str = request.form.get("date_time")
        dt = datetime.fromisoformat(date_time_str)
        hour = dt.hour
        weekday = dt.weekday()
        if weekday>=5:
            weekend = 1
        else:
            weekend = 0
        month = dt.month

        path = 'Data/SGDmodel_stable_full.pkl'

        prediction = functions_mysha.predict_availability(model_path=path, capacity=station_capacity, station_id=station_id, hour_of_day=hour, day_of_week=weekday, month=month, is_weekend=weekend)

        stations_list = functions.stations_dict('Data/station_data.csv')

        return render_template("predict.html", stations=stations_list, prediction=prediction, station_name=station_name)


