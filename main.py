from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import os
import uvicorn
import sys
from utils import extract_feature,convert
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import collections
import contextlib
import wave
import webrtcvad
import os
#from vad_audio_split import *
app = FastAPI()

model = pickle.load(open(f"{os.getcwd()}/model.model", "rb"))



@app.post("/upload_audio/")
async def root(audio: UploadFile = File(...)):
    print(audio.file)
    try:
        os.mkdir("testing")
        print(os.getcwd())
    except Exception as e:
        print(e)
    filedir = os.getcwd()+"/testing/"+audio.filename.replace(" ", "-")
    file_name= os.path.basename(filedir)

    with open(os.getcwd()+"/testing/"+file_name,'wb') as f:
        f.write(audio.file.read())
        f.close()

    print(file_name)


    wav_file_pre = f"{filedir}"
    wav_file = convert(wav_file_pre)

    x_test =extract_feature(wav_file)
    y_pred=model.predict(np.array([x_test]))

    return {"filename": file_name, "prediction": y_pred[0]}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
