# SE Coursework, 6th semester

This repo contains the source code for my 6th semester Software Engineering Coursework.

It is supposed to be a quite small app with Django-based web interface and two modules: one for storing texts and one for classifying them

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
- [x] Make clustering performance better
- [ ] Prepare test dataset
- [ ] Update web interface (grid layout and CSS)
