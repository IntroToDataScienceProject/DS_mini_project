from app import app
from datetime import  datetime
import functions
import functions
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
        station_capacity = int(request.form.get("station_capacity"))
        date_time_str = request.form.get("date_time")
        dt = datetime.fromisoformat(date_time_str)
        selected_date = dt.strftime("%-d.%-m.%Y")
        selected_time = dt.strftime("%H:%M")

        path = 'Data/SGDmodel_stable_full_tuned.pkl'

        prediction = functions.predict_availability(model_path=path, station_name=station_name, future_datetime=dt)

        functions.plot_pie_chart(available=prediction['available'], empty=prediction['empty'])

        nearby=functions.fetch_nearby_stations(station_name=station_name, capacity_csv="Data/station_data.csv")

        stations_list = functions.stations_dict('Data/station_data.csv')

        return render_template("predict.html", stations=stations_list, prediction=prediction, station_name=station_name, date=selected_date, time=selected_time, nearby=nearby)


