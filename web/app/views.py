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
