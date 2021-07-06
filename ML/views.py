from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import os
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.conf import settings
from ML.apps import Predictor
from ML import get_wav, to_mfcc, normalize_mfcc,segment_one, remove_silence, add_dim, make_segment
import numpy as np
from p_tqdm import p_map


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

    X = make_segment(X, COL_SIZE = settings.COL_SIZE, OVERLAP_SIZE = settings.OVERLAP_SIZE)
    X = p_map(to_mfcc, X)
    X = p_map(normalize_mfcc,X)
    X = p_map(add_dim, X)

    prediction = 6

    try:
        prediction = Predictor.model.predict(np.array(X))
        prediction = np.argmax(prediction, axis = 1)
        prediction = np.bincount(prediction)
        prediction = np.argmax(prediction)
    except:
        pass


    return JsonResponse({'result':int(prediction)})