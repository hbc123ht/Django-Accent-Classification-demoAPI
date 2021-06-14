from django.shortcuts import render, redirect
from django.views.decorators.csrf import csrf_exempt
import wave
import os
from ML import Accent_Identification
from ML.models import Result

result = Result()
ML = Accent_Identification()
# Create your views here.
def index(request):
    ctx = {
        'result' : result,
    }
    return render(request, 'index.html', ctx)

@csrf_exempt
def predict(request):
    uploadedFile = open("test.wav", "wb")
    # the actual file is in request.body
    uploadedFile.write(request.body)
    uploadedFile.close()
    
    result.result = ML.predict('test.wav')

    return redirect('/')

    

    return redirect('/')
