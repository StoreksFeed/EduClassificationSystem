import requests

from django.core.paginator import Paginator
from django.contrib import messages
from django.shortcuts import render, redirect
from django.http import HttpResponse

from .models import *

# Create your views here.

def healthcheck(request):
    if request.method == 'GET':
        return HttpResponse(status=200, content="{'status': 'OK'}")
    else:
        return HttpResponse(status=405, content='Method Not Allowed')

def listEntriesView(request):
    if request.method == 'GET':
        search_type = request.GET.get('searchType')
        search_query = request.GET.get('searchQuery')

        if search_query:
            if search_type == 'entry':
                entries = Entry.objects.filter(uuid=search_query).order_by('group')
            elif search_type == 'group':
                entries = Entry.objects.filter(group=search_query).order_by('group')
        else:
            entries = Entry.objects.all().order_by('group')

        paginator = Paginator(entries, 10)
        page_number = request.GET.get('page')
        page_obj = paginator.get_page(page_number)

        return render(request, 'index.html', {
            'page_obj': page_obj,
            'search_query': search_query,
            'search_type': search_type,
        })
    else:
        return HttpResponse(status=405, content='Method Not Allowed')

def addEntryView(request):
    if request.method == 'POST':
        text = request.POST.get('text')
        entry = Entry.objects.create(text=text)
        messages.success(request, f'Запись успешно добавлена! UUID: {entry.uuid}')
        return redirect('index')
    else:
        return HttpResponse(status=405, content='Method Not Allowed')

def editEntryView(request, uuid):
    if request.method == 'POST':
        entry = Entry.objects.get(uuid=uuid)
        entry.text = request.POST.get('text')
        entry.status = 3
        entry.save()
        messages.success(request, f'Запись {uuid} успешно изменена!')
        return redirect('index')
    else:
        return HttpResponse(status=405, content='Method Not Allowed')

def deleteEntryView(request, uuid):
    if request.method == 'POST':
        entry = Entry.objects.get(uuid=uuid)
        entry.delete()
        messages.success(request, f'Запись {uuid} успешно удалена!')
        return redirect('index')
    else:
        return HttpResponse(status=405, content='Method Not Allowed')

def classifyEntryView(request, uuid):
    if request.method == 'POST':
        try:
            response = requests.get(f'http://classifier:8001/classification/{uuid}')
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
            clusters = request.POST.get('clusterCount')
            if clusters:
                response = requests.get(f'http://classifier:8001/clustering/{clusters}')
            else:
                response = requests.get('http://classifier:8001/clustering/')
            
            if response.status_code == 200:
                messages.success(request, 'Кластеризация успешно завершена!')
            else:
                messages.error(request, f'Ошибка кластеризации: {response.status_code}')
        except requests.exceptions.RequestException as e:
            messages.error(request, f'Ошибка соединения: {str(e)}')

    return redirect('index')
