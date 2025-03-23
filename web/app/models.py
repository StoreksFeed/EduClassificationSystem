import uuid
from cassandra.cqlengine import columns
from django_cassandra_engine.models import DjangoCassandraModel
# Create your models here.

class Entry(DjangoCassandraModel):
    uuid = columns.UUID(primary_key=True, default=uuid.uuid4)
    text = columns.Text(required=False)
    status = columns.Integer(default=0)
    group = columns.UUID(default=lambda: uuid.UUID(int=0))

class Group(DjangoCassandraModel):
    uuid = columns.UUID(primary_key=True, default=uuid.uuid4)
    count = columns.Integer(default=0)
    vector = columns.List(columns.Float, required=False)
    