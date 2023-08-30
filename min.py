from sklearn.preprocessing import OneHotEncoder
from flask import Flask,render_template,request
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from scipy.stats import zscore
from sklearn.ensemble import StackingRegressor
from sklearn.neural_network import MLPRegressor
import seaborn as sns
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime,timedelta
from sklearn.metrics import mean_absolute_error
from functools import reduce
import pickle


pickle_out=open("pickle.pkl",'wb')
pickle.dump(model,pickle_out)
loaded_model=pickle.load(open("pickle.pkl",'rb'))
result=loaded_model.score(testX,testY)
print(result)

def unique(list1):
    ans=reduce(lambda re,x: re+[x] if x not in re else re,list1,[])
    print(ans)

def posix_time(dt):
    return (dt-datetime(1970,1,1))/timedelta(seconds=1)

n1features=[]
n2features=[]
x_scaler=MinMaxScaler()
y_scaler=MinMaxScaler()
regre=MLPRegressor(random_state=1,max_iter=500)


#USER INPUT

#ip=[0,276.900,3,7,5,2017,4,2,0]
#ip=x_scaler.transform([ip])
#out=regre.predict(ip)
#print("before inverse scaling = ",out)


#y_pred=y_scaler.inverse_transform([out])
#print("traffic volume = ",y_pred)

#if (y_pred<=1000):
 #   print("No Traffic")
#elif(y_pred>1000 and y_pred<=3000):
#    print("Busy or Normal flow")    
# elif(y_pred>3000 and y_pred<5500):
  #  print("Heavy Traffic")    
#else:
 #   print("Worst Case")   


app=Flask(__name__,static_url_path='')

@app.route("/")
def root():
    return render_template("web.html")

#importing the dataset
@app.route("/train",methods=['POST','GET'])
def train():
    data=pd.read_csv("Train.csv")
    data=data.sort_values(by=["date_time"],ascending=True).reset_index(drop=True)
    last_n_hours=[1,2,3,4,5,6]
    for n in last_n_hours:
        data[f"last_{n}_hour_traffic"]=data["traffic_volume"].shift(n)
    data=data.dropna().reset_index(drop=True)
    data.loc[data["is_holiday"]!='None','is_holiday']=1
    data.loc[data["is_holiday"]=='None','is_holiday']=0
    data["is_holiday"]=data["is_holiday"].astype(int)
    
    data["date_time"]=pd.to_datetime(data["date_time"])
    data["hour"]=data["date_time"].map(lambda x: int (x.strftime("%H")))
    data["month_day"]=data["date_time"].map(lambda x: int (x.strftime("%d")))
    data["weekday"]=data["date_time"].map(lambda x: x.weekday()+1)
    data["month"]=data["date_time"].map(lambda x: int (x.strftime("%m")))
    data["year"]=data["date_time"].map(lambda x: int (x.strftime("%Y")))
    data.to_csv("traffic_volume_data.csv",index=None)
    
    
    sns.set()
    plt.rcParams["font.sans-serif"]=["SimHei"]
    plt.rcParams["axes.unicode_minus"]=False
    warnings.filterwarnings("ignore")
    data=pd.read_csv("traffic_volume_data.csv")
    data=data.sample(9994).reset_index(drop=True)
    label_columns=["weather_type","weather_description"]
    numeric_columns=["is_holiday","temperature","weekday","hour","month_day","year","month"]
    
    
    features=numeric_columns+label_columns
    x=data[features]
    x.head()
    x.shape
    
    
    n1=data["weather_type"]
    n2=data["weather_description"]
    unique(n1)
    unique(n2)
    n1features=["Rain","Clouds","Clear","Snow","Mist","Drizzle","Haze","Thunderstorm","Fog","Smoke","Squall"]
    n2features=["Light rain","Few clouds","Sky is clear","Light snow ","Mist","Broken clouds","Moderate rain","Drizzle","Overcast clouds","Scattered clouds","Haze","Thunderstorm with heavy rain","Thunderstorm with light rain"
            "Proximity thunderstorm with rain","Thunder with drizzlle","Smoke","Thunderstorm","Proximity shower rain"]
    n11=[]
    n22=[]
    for i in range(9994):
        if (n1[i]) not in n1features:
            n11.append(0)
        else:
            n11.append((n1features.index(n1[i]))+1)
        if (n2[i]) not in n2features:
            n22.append(0)
        else:
            n22.append((n1features.index(n2[i]))+1)
    #print(n11)
    data["weather_type"]=n11
    data["weather_description"]=n22


    # DATA PREPARATION
    features=numeric_columns + label_columns
    target = ["traffic_volume"]
    X=data[features]
    y=data[target]
    x.head(6)
    
    print(data[features].hist(bins=20,))


    # features scaling
    
    X = x_scaler.fit_transform(X)
    y=y_scaler.fit_transform(y).flatten()
    warnings.filterwarnings("ignore")
    regre=MLPRegressor(random_state=1,max_iter=500)
    regre.fit(X,y)
    
    #print(X[:5])
    #VISU
    #metrics=["month","month_day","weekday","hour"]
    #fig=plt.figure(figsize=(8,4*len(metrics)))
    #for i, metric in enumerate(metrics):
    #   ax=fig.add_subplot(len(metrics),1,i+1)
    #   ax.plot(data.groupby(metric)["traffic_volume"].mean(),'-o')
    #   ax.set_xlabel(metric)
    #   ax.set_ylabel("Mean Traffic")
    #   ax.set_title(f"Traffic Trend By {metric}")
    #plt.tight_layout()    
    #plt.show()


    #Train the model
    

    
    #Error elv
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error
    tarinX,testX,trainY,testY=train_test_split(X,y,test_size=0.2)
    y_pred=regre.predict(testX)
    print("mean absolute error = ",mean_absolute_error(testY,y_pred))
    #####################

    regre=MLPRegressor(random_state=1,max_iter=500).fit(X,y)
    new=[]
    print("Predicted OutPut = ",regre.predict(X[:10]))
    print("Actual OutPut =", y[:10])

    pickle.dump(regre,open('traffic.pkl','wb'))

    return render_template("train.html")




@app.route('/predict',methods=['POST','GET'])
def predict():
    ip=[]
    if(request.form['is_holiday']=='yes'):
        ip.append(1)
    else:
        ip.append(0)

    ip.append(int(request.form['temperature']))    
    ip.append(int(request.form['day'])) 
    ip.append(int(request.form['time'][:2])) 
    D=request.form['date']
    ip.append(int(D[8:]))
    ip.append(int(D[:4]))
    ip.append(int(D[5:7]))

    s1=request.form.get('x0')
    s2=request.form.get('x1')

    if(s1) not in n1features:
        ip.append(0)
    else:
        ip.append((n1features.index(s1))+1)

    if(s2) not in n2features:
        ip.append(0)
    else:
        ip.append((n2features.index(s2))+1)    
    ip=x_scaler.transform([ip])
    out=regre.predict(ip)

    print("before inverse scaling = ",out)
    



    y_pred=y_scaler.inverse_transform([out])
    print("traffic volume = ",y_pred)
    s=''
    if (y_pred<=1000):
        print("No Traffic")
        s='No Traffic'
    elif(y_pred>1000 and y_pred<=3000):
        print("Busy or Normal flow")    
        s='Busy Or No traffic'
    elif(y_pred>3000 and y_pred<5500):
        print("Heavy Traffic")
        s='Heavy Traffic'    
    else:
        print("Worst Case")
        s="Worst Traffic"
    return render_template("train.html",datal=ip,op=y_pred,statement=s)  
  



if __name__=='__main__':
    app.run(debug=True)














 

