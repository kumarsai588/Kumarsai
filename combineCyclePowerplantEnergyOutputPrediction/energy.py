from flask import Flask,render_template,request
import joblib
import numpy as np
model=joblib.load("CCPP_linear_model.pkl")
app=Flask(__name__)
@app.route('/')
def h():
    return render_template("energy.html")
@app.route('/Home',methods=["POST"])
def Home():
    data1=request.form['tem']
    data2=request.form['ex']
    data3=request.form['Am']
    data4=request.form['hum']
    arr=np.array([[data1,data2,data3,data4]],dtype=float)
    pred=model.predict(arr)
    return render_template('energy.html',data="THE ENERGY-OUTPUT IS : {}".format(pred))
if(__name__=="__main__"):
    app.run(debug=True)
