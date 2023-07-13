from flask import Flask , render_template,request
import pickle
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler

model = pickle.load(open('model.pkl','rb'))
ms = pickle.load(open('ms.pkl','rb'))

#mod = joblib.load('model_joblib')

app = Flask(__name__)
#model = pickle.load(open('model.pkl','rb')
@app.route("/")
def hello_world():
    return render_template("index.html")
    #return("Akash Dipak Aaglawe")
     
@app.route('/predict',methods=['post'])
def prdict_placement():
    
    n=int(request.form.get('n'))
    #end_date=request.form.get('end date')
    
    
    # prediction
    
    d=[[0.4176929748482222, 0.40598438855160446, 0.4024284475281873, 0.39869904596704264]]
    
    l=[]
    da=[]
    result=[]
    for i in range(n):
        m = []
        if(len(d)<n):
            m.append(d[i][1])
            m.append(d[i][2])
            m.append(d[i][3])
            a=(model.predict([d[i]]))
            result.append(a[0])
            m.append(a[0])
            d.append(m)
            #print(d)
            #n=n-1

        else:
            
            a=(model.predict([d[i]]))
            result.append(a[0])
            break
        
        
    #return(str(result))
    result = np.array(result)
    result=result.reshape(-1,1)
    re=[]
    for i in result:
        
        result = ms.inverse_transform(result)
        r=(ms.inverse_transform([i]))
        re.append(r[0][0])
    return render_template('index.html',re=re)
    
    



if __name__=="__main__":
    app.run(debug=True,port = 8000)
    