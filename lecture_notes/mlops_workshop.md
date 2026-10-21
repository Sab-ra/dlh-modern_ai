# End-to-End MLOps Workshop 

From Model Training to Local Deployment with MLflow, FastAPI, and Docker | Mac handbook 

1. Objective and terminology 

Train multiple models, log MLflow runs, select on validation macro F1, export a trusted model, expose a FastAPI HTTP endpoint, package it with Docker, and test predictions locally. Conda isolates Python dependencies; MLflow tracks experiments; FastAPI defines API endpoints; Uvicorn runs the server; Dockerfile is a recipe, image is a package, container is a running instance. 

2. Install and verify Docker 

Install Anaconda/Miniconda and Docker Desktop (Apple Silicon or Intel). Start Docker Desktop; wait for engine startup. macOS and Ubuntu share most shell commands but not Docker installation. 

```bash
open -a Docker 
docker --version 
docker info 
docker run hello-world
```
 

Expected panel/output: Hello from Docker! If daemon connection fails, start Docker Desktop. hello-world exits after printing; this is normal. 

3. Conda and Python packages 

```bash
mkdir -p ~/mlops-workshop 
cd ~/mlops-workshop 
conda create -n mlops-workshop python=3.11 -y 
conda activate mlops-workshop
```

```bash
python -m pip install scikit-learn==1.9.1 numpy==2.4.6 joblib==1.6.0 mlflow==3.17.0 fastapi==0.143.0 uvicorn==0.54.0 
python -c 'import sklearn, mlflow, fastapi; print("MLOps environment ready")'
```

Explanation: `mkdir/New-Item` creates a folder; `cd/Set-Location` enters it; `conda create` makes Python 3.11 environment; `activate` selects it; `pip` installs dependencies; the final command verifies imports. In Bash, ! inmlside double quotes may trigger history expansion. 

4. Files and model training (train.py) 

Unzip companion code ZIP into the project folder. `train.py` creates `model/best_model.pkl`, `model/metadata.json` and `mlflow.db`. Full code: 

```python
import json, joblib, mlflow, mlflow.sklearn 
from pathlib import Path 
from sklearn.datasets import load_wine 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler 
from sklearn.pipeline import make_pipeline 
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.svm import SVC 
from sklearn.metrics import accuracy_score, f1_score 
root=Path(__file__).resolve().parent 
mlflow.set_tracking_uri('sqlite:///'+str(root/'mlflow.db')) 
mlflow.set_experiment('Wine Classification Workshop') 
w=load_wine() 
Xdev,Xtest,ydev,ytest=train_test_split(w.data,w.target,test_size=.2,stratify=w.target,random_state=42) 
Xtr,Xval,ytr,yval=train_test_split(Xdev,ydev,test_size=.25,stratify=ydev,random_state=42) 
models={'Logistic Regression':make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000)), 'Random Forest':RandomForestClassifier(n_estimators=100,random_state=42),'SVM':make_pipeline(StandardScaler(),SVC())} 
best,score,name=None,-1,None 
for label,m in models.items(): 
 with mlflow.start_run(run_name=label): 
  m.fit(Xtr,ytr); pred=m.predict(Xval) 
  a=accuracy_score(yval,pred); f=f1_score(yval,pred,average='macro') 
  mlflow.log_param('model_name',label);mlflow.log_metric('validation_accuracy',a);mlflow.log_metric('validation_f1_macro',f) 
  mlflow.sklearn.log_model(sk_model=m,name='model',serialization_format='cloudpickle') 
  print(label,'accuracy',round(a,4),'F1',round(f,4)) 
  if f>score:best,score,name=m,f,label 
best.fit(Xdev,ydev);test_pred=best.predict(Xtest) 
(root/'model').mkdir(exist_ok=True) 
joblib.dump(best,root/'model/best_model.pkl') 
meta={'model_name':name,'validation_f1_macro':score,'test_f1_macro':f1_score(ytest,test_pred,average='macro'),'feature_names':list(w.feature_names),'class_names':list(w.target_names)} 
(root/'model/metadata.json').write_text(json.dumps(meta,indent=2)) 
print('Best:',name,'test F1:',round(meta['test_f1_macro'],4)) 
```

Explanation: Wine data has 178 rows, 13 features, 3 classes. St
ratified splitting reserves a held-out test set and validation subset. Pipelines scale Logistic Regression and SVM without leakage; Random Forest is unscaled. Each MLflow run logs validation accuracy/F1 and a model artifact. The highest validation macro F1 is selected, refitted on development data, evaluated once on test data, and saved. This corrects the earlier test-set-selection demonstration, so metrics may differ. 

`MLflow 3.17` uses `SQLite` because filesystem tracking is in maintenance mode. `cloudpickle` bypasses `skops` tree-type warnings but MUST only be loaded from trusted sources. 

```bash
python train.py
``` 

5. MLflow dashboard 

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000 
```

Open py. Expected panel: Experiments sidebar → Wine Classification Workshop → runs table → click a model run → Metrics, Parameters, Artifacts. Compare validation_f1_macro. Ctrl+C stops the server; it does not delete experiments. Use a second terminal for FastAPI. 

6. Serving code (app.py) 

import json, joblib, numpy as np 
from pathlib import Path 
from fastapi import FastAPI, HTTPException 
from pydantic import BaseModel, Field 
root=Path(__file__).resolve().parent 
model=joblib.load(root/'model/best_model.pkl') 
meta=json.loads((root/'model/metadata.json').read_text()) 
app=FastAPI(title='Wine Classification API',version='1.0') 
class WineInput(BaseModel): 
 features:list[float]=Field(min_length=13,max_length=13) 
@app.get('/') 
def home():return {'status':'running'} 
@app.get('/model-info') 
def info():return meta 
@app.post('/predict') 
def predict(data:WineInput): 
 x=np.array(data.features).reshape(1,-1) 
 if not np.isfinite(x).all():raise HTTPException(422,'Features must be finite') 
 y=int(model.predict(x)[0]) 
 return {'predicted_class':y,'class_name':meta['class_names'][y],'model':meta['model_name']} 
 

Explanation: joblib.load reads the saved trusted model; metadata supplies class labels. app=FastAPI() creates the service. @app.get registers status/model-info endpoints. @app.post("/predict") registers a POST API. Pydantic validates exactly 13 features. NumPy reshapes them into one row; model.predict returns a class; the endpoint responds with JSON. Invalid input produces HTTP 422. 

python -m uvicorn app:app --reload --port 8000 
 

Open http://localhost:8000/docs. Swagger UI panel: GET /, GET /model-info, POST /predict. Expand POST /predict → Try it out → edit Request body → Execute. A 200 response with predicted_class means success. The response example is NOT the input editor. 

{"features":[14.23,1.71,2.43,15.6,127,2.80,3.06,0.28,2.29,5.64,1.04,3.92,1065]} 
 

7. Docker image files and explanation 

requirements.txt lists only serving dependencies (MLflow is not needed in the image): 

fastapi==0.143.0 
uvicorn==0.54.0 
scikit-learn==1.9.1 
numpy==2.4.6 
joblib==1.6.0 
 

Dockerfile contains build instructions: 

FROM python:3.11-slim 
WORKDIR /app 
COPY requirements.txt . 
RUN pip install --no-cache-dir -r requirements.txt 
COPY app.py . 
COPY model/ ./model/ 
EXPOSE 8000 
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"] 
 

FROM selects Linux/Python; WORKDIR sets /app; COPY adds files; RUN installs packages; EXPOSE documents port 8000; CMD starts Uvicorn listening on all container interfaces. 

.dockerignore excludes development files and the MLflow database: 

__pycache__ 
*.pyc 
mlflow.db 
mlruns 
.git 
 

8. Build, run and verify container 

docker build -t wine-classifier:v1 . 
docker run --rm -p 8000:8000 --name wine-api wine-classifier:v1 
 

Stop the earlier local Uvicorn with Ctrl+C first; otherwise port 8000 is occupied. docker build creates a reusable image. docker run creates a container; -p maps port 8000; --rm removes the stopped container, not the image. Open /docs again and send the same request. 

docker ps 
docker logs wine-api 
docker exec wine-api python -c "import os; print(os.path.exists('/.dockerenv'))" 
docker stop wine-api 
 

Expected: docker ps lists wine-api; logs show POST /predict 200 OK; the container check prints True. Stop and refresh /docs: it should be unavailable unless another server is listening. This proves serving came from Docker, although the UI looks identical. 

9. Troubleshooting and classroom checks 

“uvicorn: command not found”: activate conda and use python -m uvicorn. “Cannot connect to Docker daemon”: start Docker Desktop. “Port allocated”: stop local Uvicorn or existing container. “Model not found”: run train.py before building. “Filesystem tracking backend”: use sqlite:///mlflow.db. “skops untrusted tree”: use the supplied cloudpickle option only for self-trained artifacts. The MLflow demo tracing warnings are not necessarily fatal. After editing code/model, rebuild the image. 

Checkpoint sequence: hello-world succeeds → three MLflow model runs appear → model files exist → FastAPI returns 200 → Docker image builds → docker ps lists wine-api → Docker logs show POST 200 → stopping container stops the service. 

10. Teaching flow and Ubuntu note 

Suggested 4 hours: setup 30 min; training+MLflow 45; FastAPI 45; Docker 60; tests/troubleshooting 40; exercises 20. Optional exercise: modify app.py response, rebuild as v2, restart, and verify version change. 

Ubuntu: Python, FastAPI, MLflow and Docker commands are largely the same as macOS after installation. Docker Engine installation differs (Linux packages/systemd), and users may need sudo or docker group access. Windows uses PowerShell/WSL2, and curl may be an alias; use Swagger UI or curl.exe. 