"""Simple next-hour indoor-temperature prediction experiment."""
import sys, os
sys.path.append(os.path.dirname(__file__))
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from simulation import simulate

FEATURES=['T_in','T_mass','T_out','Q_solar','Q_internal','Q_HVAC']

def prepare(df):
    d=df.copy(); d['T_next']=d.T_in.shift(-1); d=d.dropna()
    return d[FEATURES],d.T_next

def evaluate(y,p):
    return {'MAE':mean_absolute_error(y,p),'RMSE':np.sqrt(mean_squared_error(y,p)),'R2':r2_score(y,p)}

def run(hours=720):
    X,y=prepare(simulate(hours))
    n=len(X); a=int(.70*n); b=int(.85*n)
    Xtr,Xte=X.iloc[:a],X.iloc[b:]; ytr,yte=y.iloc[:a],y.iloc[b:]
    models={'Linear Regression':LinearRegression(), 'Random Forest':RandomForestRegressor(n_estimators=150,max_depth=10,random_state=42,n_jobs=1)}
    results={'Persistence':evaluate(yte,Xte.T_in.values)}
    predictions={}
    for name,m in models.items():
        m.fit(Xtr,ytr); p=m.predict(Xte); results[name]=evaluate(yte,p); predictions[name]=p
    return results,yte,predictions

if __name__=='__main__':
    r,_,_=run();
    for k,v in r.items(): print(k, {m:round(x,3) for m,x in v.items()})
