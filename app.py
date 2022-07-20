from flask import *
import model
import datetime
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('home.html')

@app.route('/predict/')
def predict():
    t_date=datetime.datetime.today().date()
    labels_con,values_con=model.predict_confirm_cases(t_date)
    labels_re,values_re=model.predict_recoverd_cases(t_date)
    labels_de,values_de=model.predict_deceased_cases(t_date)
    return render_template("prediction.html",  values_con=values_con,values_de=values_de,values_re=values_re,labels=labels_con)



if __name__ == '__main__':
    app.run()
