from django.urls import path

from .views import *

urlpatterns = [
    path('', listEntriesView, name='index'),
    path('entry/add/', addEntryView, name='addEntry'),
    path('entry/edit/<uuid:uuid>/', editEntryView, name='editEntry'),
    path('entry/delete/<uuid:uuid>/', deleteEntryView, name='deleteEntry')
]
