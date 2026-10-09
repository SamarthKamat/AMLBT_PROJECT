"""REFERENCE ONLY: train and persist the seven domain reference model sets.

Nothing at runtime loads these artifacts. Uploaded datasets always get models trained from scratch by ml_engine.py.
The artifacts exist to document the seven bundled demo datasets and to benchmark the dynamic pipeline
(see reference_models.py --validate and tests/test_reference_models.py). Retraining is optional:
    python train_models.py

Artifacts are named <domain>_<task>_<algorithm>.joblib so their domain and purpose are
immediately identifiable. Runtime analysis of uploaded datasets is performed dynamically
by ml_engine.py; these artifacts are reproducible domain reference models.
"""
from pathlib import Path
import json, shutil
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, f1_score, roc_auc_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier, IsolationForest
from xgboost import XGBRegressor, XGBClassifier
from config import SAMPLE_DIR, MODEL_DIR, DOMAINS

TARGETS={
 'company':('Profit','Risk_Flag','Demand'), 'hospital':('Patients','Capacity_Risk','Patients'), 'college':('Attendance','Attendance_Risk','Attendance'),
 'municipality':('Waste_Collected','Service_Risk','Complaints'), 'ngo':('Beneficiaries','Coverage_Risk','Beneficiaries'), 'community':('Resource_Requests','Shortage_Risk','Resource_Requests'), 'event_organizer':('Attendance','Capacity_Risk','Attendance')}

def prep(X):
    nums=X.select_dtypes(include=np.number).columns.tolist(); cats=[c for c in X.columns if c not in nums]
    return ColumnTransformer([('num',Pipeline([('imputer',SimpleImputer(strategy='median')),('scale',StandardScaler())]),nums),('cat',Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore'))]),cats)],remainder='drop')

def train_domain(domain):
    df=pd.read_csv(SAMPLE_DIR/f'{domain}.csv'); reg_target,cls_target,forecast_target=TARGETS[domain]
    out=MODEL_DIR/domain; out.mkdir(parents=True,exist_ok=True)
    for old in out.glob('*.joblib'): old.unlink()
    work=df.copy()
    X=work.drop(columns=[reg_target,cls_target,forecast_target],errors='ignore').drop(columns=['Date'],errors='ignore')
    y=pd.to_numeric(work[reg_target],errors='coerce'); mask=y.notna(); X=X.loc[mask]; y=y.loc[mask]
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42)
    reg_candidates={
      'linear':LinearRegression(),
      'decision_tree':DecisionTreeRegressor(max_depth=8,min_samples_leaf=3,random_state=42),
      'random_forest':RandomForestRegressor(n_estimators=300,max_depth=12,min_samples_leaf=2,random_state=42,n_jobs=2),
      'gradient_boosting':GradientBoostingRegressor(n_estimators=250,max_depth=3,learning_rate=.04,random_state=42),
      'xgboost':XGBRegressor(n_estimators=300,max_depth=5,learning_rate=.04,subsample=.85,colsample_bytree=.85,objective='reg:squarederror',random_state=42,n_jobs=2),
    }
    reg_scores={}
    for name,est in reg_candidates.items():
        pipe=Pipeline([('prep',prep(X)),('model',est)]); pipe.fit(Xtr,ytr); p=pipe.predict(Xte)
        reg_scores[name]={'R2':float(r2_score(yte,p)),'MAE':float(mean_absolute_error(yte,p)),'RMSE':float(mean_squared_error(yte,p)**.5)}
        joblib.dump(pipe,out/f'{domain}_regression_{name}.joblib', compress=('lzma',9))
    cy=work.loc[mask,cls_target].astype(str); labels=LabelEncoder(); ycls=labels.fit_transform(cy); cX=X.copy(); cxtr,cxte,cytr,cyte=train_test_split(cX,ycls,test_size=.2,random_state=42,stratify=ycls)
    nclasses=len(labels.classes_)
    clf_candidates={
      'logistic':LogisticRegression(max_iter=1000),
      'decision_tree':DecisionTreeClassifier(max_depth=8,min_samples_leaf=3,random_state=42),
      'random_forest':RandomForestClassifier(n_estimators=300,max_depth=12,min_samples_leaf=2,random_state=42,n_jobs=2),
      'gradient_boosting':GradientBoostingClassifier(n_estimators=250,max_depth=3,learning_rate=.04,random_state=42),
      'xgboost':XGBClassifier(n_estimators=300,max_depth=4,learning_rate=.05,subsample=.85,colsample_bytree=.85,eval_metric='logloss',random_state=42,n_jobs=2),
    }
    clf_scores={}
    for name,est in clf_candidates.items():
        pipe=Pipeline([('prep',prep(cX)),('model',est)]); pipe.fit(cxtr,cytr); cp=pipe.predict(cxte); proba=pipe.predict_proba(cxte)
        auc=float(roc_auc_score(cyte,proba[:,1])) if nclasses==2 else None
        clf_scores[name]={'Accuracy':float(accuracy_score(cyte,cp)),'F1':float(f1_score(cyte,cp,average='weighted')),'ROC_AUC':auc}
        joblib.dump({'pipeline':pipe,'classes':labels.classes_.tolist(),'target':cls_target},out/f'{domain}_classification_{name}.joblib', compress=('lzma',9))
    # Forecast reference model
    dates=pd.to_datetime(work['Date'],errors='coerce'); series=work.assign(Date=dates).dropna(subset=['Date']).groupby('Date')[forecast_target].mean().sort_index().asfreq('D').interpolate().dropna(); vals=series.values.astype(float); lags=7
    FX=np.array([vals[i-lags:i] for i in range(lags,len(vals))]); Fy=vals[lags:]; fmodel=RandomForestRegressor(n_estimators=300,max_depth=10,min_samples_leaf=2,random_state=42,n_jobs=2); fmodel.fit(FX,Fy); joblib.dump(fmodel,out/f'{domain}_forecasting_random_forest.joblib', compress=('lzma',9))
    iso_cols=work.select_dtypes(include=np.number).columns.tolist(); iso=IsolationForest(n_estimators=300,contamination=.03,random_state=42); iso.fit(work[iso_cols].fillna(work[iso_cols].median())); joblib.dump(iso,out/f'{domain}_anomaly_isolation_forest.joblib', compress=('lzma',9))
    best_reg=max(reg_scores,key=lambda k:reg_scores[k]['R2']); best_cls=max(clf_scores,key=lambda k:clf_scores[k]['F1'])
    metadata={'domain':domain,'regression_target':reg_target,'classification_target':cls_target,'forecast_target':forecast_target,'rows':len(df),'training_feature_columns':X.columns.tolist(),'classification_feature_columns':cX.columns.tolist(),'regression_candidates':reg_scores,'classification_candidates':clf_scores,'selected_regression':f'{domain}_regression_{best_reg}.joblib','selected_classification':f'{domain}_classification_{best_cls}.joblib','forecasting_model':f'{domain}_forecasting_random_forest.joblib','anomaly_model':f'{domain}_anomaly_isolation_forest.joblib','anomaly_features':iso_cols}
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf8')
    print(domain,'regression=',best_reg,'classification=',best_cls)

if __name__=='__main__':
    for domain in DOMAINS: train_domain(domain)
