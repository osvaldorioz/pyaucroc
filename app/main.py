from fastapi import FastAPI
import auc_roc_module
import time
import json
from pydantic import BaseModel
from typing import List

app = FastAPI()

# Definir el modelo para la matriz
class ScoresDim(BaseModel):
    values: List[float]

class LabelsDim(BaseModel):
    values: List[int]

@app.post("/auc-roc")
async def ac(labels: LabelsDim, scores: ScoresDim):
   
    start = time.time()

    auc = auc_roc_module.calculate_auc(labels.values, scores.values)

    end = time.time()

    var1 = 'Time taken in seconds: '
    var2 = end - start

    str = f'{var1}{var2}'.format(var1=var1, var2=var2)
    
    data = {
        "AUC-ROC": auc,
        "Time taken": str
    }
    jj = json.dumps(data)
    
    return jj