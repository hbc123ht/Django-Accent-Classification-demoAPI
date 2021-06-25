from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import os
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.conf import settings
from ML.apps import Predictor
from ML import get_wav, to_mfcc, normalize_mfcc,segment_one, remove_silence
import numpy as np

# Create your views here.
def index(request):
    return render(request, 'index.html')

@csrf_exempt
def predict(request):

    audio_data = request.FILES['audio']
    default_storage.save('voice.wav', ContentFile(audio_data.read()))
    
    X = get_wav('voice.wav')

    os.remove('voice.wav')
    X = remove_silence(X)

    X = to_mfcc(X)

    X = normalize_mfcc(X)

    X = segment_one(X)


    prediction = 6
    try:
        prediction = Predictor.model.predict(X)
        prediction = np.argmax(prediction, axis = 1)
        prediction = np.bincount(prediction)
        prediction = np.argmax(prediction)
    except:
        pass


    return JsonResponse({'result':int(prediction)})