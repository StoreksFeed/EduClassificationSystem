import requests

from django.contrib import messages
from django.shortcuts import render, redirect
from django.http import HttpResponse

from .models import *

# Create your views here.

def listEntriesView(request):
    if request.method == 'GET':
        entries = Entry.objects.all()
        return render(request, 'index.html', {'entries': entries})
    else:
        return HttpResponse(status=405, content='Method Not Allowed')

def addEntryView(request):
    if request.method == 'GET':
        return render(request, 'addEntry.html')
    elif request.method == 'POST':
        text = request.POST.get('text')
        Entry.objects.create(text=text)
        return redirect('index')
    else:
        return HttpResponse(status=405, content='Method Not Allowed')

def editEntryView(request, uuid):
    if request.method == 'GET':
        entry = Entry.objects.get(uuid=uuid)
        return render(request, 'editEntry.html', {'entry': entry})
    elif request.method == 'POST':
        entry = Entry.objects.get(uuid=uuid)
        entry.text = request.POST.get('text')
        entry.save()
        return redirect('index')
    else:
        return HttpResponse(status=405, content='Method Not Allowed')

def deleteEntryView(request, uuid):
    if request.method == 'POST':
        entry = Entry.objects.get(uuid=uuid)
        entry.delete()
        return redirect('index')
    else:
        return HttpResponse(status=405, content='Method Not Allowed')

def classifyEntryView(request, uuid):
    if request.method == 'POST':
        try:
            response = requests.post(f'http://classifier:8010/classification/{uuid}')
            if response.status_code == 200:
                messages.success(request, 'Классификация успешно завершена!')
            else:
                messages.error(request, f'Ошибка классификации: {response.status_code}')
        except requests.exceptions.RequestException as e:
            messages.error(request, f'Ошибка соединения: {str(e)}')

    return redirect('index')

def clusterizeView(request):
    if request.method == 'POST':
        try:
            response = requests.post('http://classifier:8010/clustering')
            if response.status_code == 200:
                messages.success(request, 'Кластеризация успешно завершена!')
            else:
                messages.error(request, f'Ошибка кластеризации: {response.status_code}')
        except requests.exceptions.RequestException as e:
            messages.error(request, f'Ошибка соединения: {str(e)}')

    return redirect('index')
