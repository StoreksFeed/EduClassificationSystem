from django.urls import path

from .views import *

urlpatterns = [
    path('', listEntriesView, name='index'),
    path('entry/add/', addEntryView, name='addEntry'),
    path('entry/edit/<uuid:uuid>/', editEntryView, name='editEntry'),
    path('entry/delete/<uuid:uuid>/', deleteEntryView, name='deleteEntry'),
    path('clusterize/', clusterizeView, name='clusterise'),
    path('entry/classfy/<uuid:uuid>/', classifyEntryView, name='classifyEntry')
]
