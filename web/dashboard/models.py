import uuid
from cassandra.cqlengine import columns
from django_cassandra_engine.models import DjangoCassandraModel
# Create your models here.

class Entry(DjangoCassandraModel):
    STATUS_OPTIONS = {
        0: "Новый",
        1: "Кластеризован",
        2: "Классифицирован",
        3: "Обновлён"
    }

    uuid = columns.UUID(primary_key=True, default=uuid.uuid4)
    text = columns.Text(required=False)
    status = columns.Integer(default=0)
    group = columns.UUID(default=lambda: uuid.UUID(int=0))

    def get_status_display(self):
        return self.STATUS_OPTIONS.get(self.status, "Неизвестно")
    
    def get_group_display(self):
        return "Отсутствует" if self.group == uuid.UUID(int=0) else self.group

class Group(DjangoCassandraModel):
    uuid = columns.UUID(primary_key=True, default=uuid.uuid4)
    count = columns.Integer(default=0)
    vector = columns.List(columns.Float, required=False)
    