from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import os
from ML import Accent_Identification
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile

ML = Accent_Identification()
# Create your views here.
def index(request):
    return render(request, 'index.html')

@csrf_exempt
def predict(request):
    # uploadedFile = open("test.wav", "wb")
    # uploadedFile.write(request.body)
    # uploadedFile.close()
    # the actual file is in request.body

    audio_data = request.FILES['audio']
    default_storage.save('voice.wav', ContentFile(audio_data.read()))
    result = None
    try:
        result = ML.predict('voice.wav')
    except:
        pass
    os.remove('voice.wav')
    return JsonResponse({'result':result})