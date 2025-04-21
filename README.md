# SE Coursework, 6th semester

This repo contains the source code for my 6th semester Software Engineering Coursework.

It is a Docker Compose project with 3 containers:
- Django based web-app for user interaction,
- database (Apache Cassandra),
- and a clustering/classification module for text.

## Build and run

```
docker-compose up --build
```

## TODO:

- [x] Init Django app
- [x] Init Cassandra
- [x] Pack it all in Docker containers
> It seems like cassandra-driver reaaaaally struggles to work under Windows
> instead opting for crashing any Python app without any logs whatsoever.
> After a day of painful debugging I decided that running it under Linux
> sounds like a quite good idea, so here it is
- [x] Add an ability to manage Cassandra tables via Django
- [x] Create all the models required
- [x] Implement clustering (for all entries)
- [x] Implement classification (for single entry based on created groups)
- [x] Add Django buttons to trigger classifier
- [x] Prepare test dataset
- [x] Update web interface (grid layout and CSS)
> Now it looks much better and supports dark mode
- [x] Check options in sklearn.cluster
> Base and advanced text preprocessing, AgglomerativeClustering and KMeans
- [x] Measure clustering perfomance based on a true headers
- [x] Compare against Lingo, STC and Random distribution
> Some nice plots can be found in the Jupyter notebook (results/notebook.ipynb)
