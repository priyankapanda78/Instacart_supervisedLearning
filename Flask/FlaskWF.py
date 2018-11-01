import flask
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

instacart = pd.read_csv("Instacartdf1.csv", header=None)
instacart.columns=['key','product_id', 'user_id', 'total_order_products', 'last_order',
       'reordered', 'days_since_prior_order', 'order_dow', 'order_hour_of_day',
       'tot_items', 'tot_orders', 'avg_order_size', 'total', 'reord_count',
       'reord_rate', 'product_name', 'aisle_id', 'department_id_2',
       'department_id_3', 'department_id_4', 'department_id_5',
       'department_id_6', 'department_id_7', 'department_id_8',
       'department_id_9', 'department_id_10', 'department_id_11',
       'department_id_12', 'department_id_13', 'department_id_14',
       'department_id_15', 'department_id_16', 'department_id_17',
       'department_id_18', 'department_id_19', 'department_id_20',
       'department_id_21', 'prod_cart', 'total_prod_cart', 'avg_cart_order',
       'user_prod_reord', 'total_ord_by_user', 'times_reord_by_user']

X = instacart.drop(['key','aisle_id','last_order','product_id','total_order_products','user_id','reordered','product_name'],axis=1)
Y = instacart['reordered']

scaler=StandardScaler()
X_std=scaler.fit_transform(X)
PREDICTOR = LogisticRegression(class_weight={1 : 7, 0 : 1}, C=1).fit(X_std,Y)

#---------- URLS AND WEB PAGES -------------#

# Initialize the app
app = flask.Flask(__name__)

# Homepage
@app.route("/")
def viz_page():
   """
   Homepage: serve our visualization page
   """
   with open("test1.html", 'r') as viz_file:
       return viz_file.read()
    
@app.route("/api", methods=["POST"])
def make_predict():
    
    data = flask.request.json
    user = data["user_id"]
    product = data["product_id"]
    print(user)
    print(product)
    
    record = instacart[(instacart.user_id== user) & (instacart.product_id == product)]
    
    print(record)
  
    #data = request.get_json(force=True)
    x = np.matrix([
              record.iloc[0]['days_since_prior_order'],
              record.iloc[0]['order_dow'],record.iloc[0]['order_hour_of_day'],
              record.iloc[0]['tot_items'], record.iloc[0]['tot_orders'], record.iloc[0]['avg_order_size'],
              record.iloc[0]['total'],record.iloc[0]['reord_count'],record.iloc[0]['reord_rate'],
              record.iloc[0]['department_id_2'],record.iloc[0]['department_id_3'],
              record.iloc[0]['department_id_4'],record.iloc[0]['department_id_5'],
              record.iloc[0]['department_id_6'],record.iloc[0]['department_id_7'],
              record.iloc[0]['department_id_8'],record.iloc[0]['department_id_9'],
              record.iloc[0]['department_id_10'],record.iloc[0]['department_id_11'],
              record.iloc[0]['department_id_12'],record.iloc[0]['department_id_13'],
              record.iloc[0]['department_id_14'],record.iloc[0]['department_id_15'],
              record.iloc[0]['department_id_16'],record.iloc[0]['department_id_17'],
              record.iloc[0]['department_id_18'],record.iloc[0]['department_id_19'],
              record.iloc[0]['department_id_20'],record.iloc[0]['department_id_21'],
              record.iloc[0]['prod_cart'],record.iloc[0]['total_prod_cart'],
              record.iloc[0]['avg_cart_order'],record.iloc[0]['user_prod_reord'],
              record.iloc[0]['total_ord_by_user'],record.iloc[0]['times_reord_by_user']
             ])

    repurchase = PREDICTOR.predict(x)
    # Put the result in a nice dict so we can send it as json
    results = {"repurchase": int(repurchase[0])}
    return flask.jsonify(results)

#--------- RUN WEB APP SERVER ------------#

# Start the app server on port 80
# (The default website port)
app.run(host='0.0.0.0')
app.run(debug=True)








