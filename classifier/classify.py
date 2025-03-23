from collections import defaultdict
from http.server import HTTPServer, BaseHTTPRequestHandler
import re
import string
import sys
import uuid

from cassandra.cluster import Cluster

import numpy as np

import torch
from transformers import BertTokenizer, BertModel

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity


def fprint(text, length=80):
    padding = (length - len(text) - 2) // 2
    print(f"{'-' * padding} {text} {'-' * padding}")


def clean_text(text):
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def get_global_model(model_name):
    if model_name not in globals():
        globals()[model_name] = [
            BertTokenizer.from_pretrained(model_name),
            BertModel.from_pretrained(model_name),
        ]
    return globals()[model_name]


def get_bert_embeddings(texts, model_name='DeepPavlov/rubert-base-cased'):
    tokenizer, model = get_global_model(model_name)
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    return torch.mean(outputs.last_hidden_state, dim=1).detach().numpy()


def best_kmeans_n_clusters(embeddings):
    range_n_clusters = list(range(2, len(embeddings) - 1))
    best_n_clusters = 2
    best_silhouette_score = -1
    for n_clusters in range_n_clusters:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings)
        silhouette_avg = silhouette_score(embeddings, cluster_labels)
        print(f'For n_clusters = {n_clusters}, the average score is : {silhouette_avg}')
        if silhouette_avg > best_silhouette_score:
            best_silhouette_score = silhouette_avg
            best_n_clusters = n_clusters
    print(f'Optimal number of clusters: {best_n_clusters}')
    return best_n_clusters

def clustering():
    cluster = Cluster(['db'])
    session = cluster.connect('coursework')
    entries = session.execute('SELECT * FROM entry')
    entries = [(entry.uuid, entry.text) for entry in entries]
    embeddings = np.array(list(get_bert_embeddings(clean_text(entry[1])) for entry in entries)).reshape((-1, 768))

    # Perform K-Means clustering with the optimal number of clusters
    fprint('K-Means Clustering')
    best_n_clusters = best_kmeans_n_clusters(embeddings)
    kmeans = KMeans(n_clusters=best_n_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)

    # Generate UUIDs for each label
    label_to_uuid = {label: uuid.uuid4() for label in labels}
    uuid_labels = [label_to_uuid[label] for label in labels]

    # Group texts based on clustering labels
    grouped_texts = defaultdict(list)
    grouped_embeddings = defaultdict(list)
    for uuid_label, row_uuid, row_embedding in zip(uuid_labels, [entry[0] for entry in entries], embeddings):
        grouped_texts[uuid_label].append(row_uuid)
        grouped_embeddings[uuid_label].append(row_embedding)

    fprint('Grouped Texts')
    session.execute('TRUNCATE group')
    # Print grouped texts
    for uuid_label, group in grouped_texts.items(): # grouped is with uuids in labels
        print(f'Group {uuid_label}:')
        for entry_uuid in group:
            print(f'  - {entry_uuid}')
            session.execute(f"UPDATE entry SET group = {uuid_label}, status = 1 WHERE uuid = {entry_uuid}")

    # Seems to not work properly. Yet.
    common_vectors = {}
    for uuid_label, embeddings in grouped_embeddings.items(): # grouped is with uuids in labels
        common_vectors[uuid_label] = np.mean(embeddings, axis=0).tolist()
        session.execute(f"INSERT INTO group (uuid, count, vector) VALUES (%s, %s, %s)", (uuid_label, len(embeddings), common_vectors[uuid_label]))

    cluster.shutdown()

def classification(entry_uuid):
    cluster = Cluster(['db'])
    session = cluster.connect('coursework')
    entry = session.execute(f'SELECT * FROM entry WHERE uuid = {entry_uuid}').one()
    entry = (entry.uuid, entry.text)

    embedding = np.array(list(get_bert_embeddings(clean_text(entry[1])))).reshape((-1, 768))

    groups = session.execute('SELECT * FROM group')
    groups = [(group.uuid, group.count, np.array(group.vector)) for group in groups]
    similarity = cosine_similarity(embedding, [group[2] for group in groups])
    for sim, group, count, vector in zip(similarity[0], [group[0] for group in groups], [group[1] for group in groups], [group[2] for group in groups]):
        if sim >= 0.9:
            print(f'{entry_uuid} -> Group {group}')
            session.execute(f"UPDATE entry SET group = {group}, status = 2 WHERE uuid = {entry_uuid}")
            session.execute(f"UPDATE group SET count = {count + 1} WHERE uuid = {group}")
            new_vector = (embedding[0] + count * vector) / (count + 1)
            session.execute(f"UPDATE group SET vector = {new_vector.tolist()} WHERE uuid = {group}")
            break
    else:
        print(f'{entry_uuid} -> No group found')
        group = str(uuid.uuid4())
        session.execute(f"INSERT INTO group (uuid, count, vector) VALUES (%s, %s, %s)", (group, 1, embedding[0].tolist()))
        session.execute(f"UPDATE entry SET group = {group} WHERE uuid = {uuid}")

    cluster.shutdown()


class RequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/plain')
        self.end_headers()
        self.wfile.write('OK'.encode())

    def do_POST(self):
        if self.path.startswith('/clustering'):
            self.send_response(200)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            clustering()
            self.wfile.write(f'Clustering completed'.encode())

        elif self.path.startswith('/classification/'):
            uuid = self.path.split('/')[-1]
            self.send_response(200)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            classification(uuid)
            self.wfile.write(f'Classification completed for UUID: {uuid}'.encode())

        else:
            self.send_response(404)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write('Invalid endpoint'.encode())

def runserver(port):
    httpd = HTTPServer(('', port), RequestHandler)
    print(f'Starting httpd server on port {port}')
    httpd.serve_forever()

if __name__ == '__main__':
    if sys.argv[1] == 'download_model':
        get_global_model(sys.argv[2])
    elif sys.argv[1] == 'runserver':
        runserver(int(sys.argv[2]))
    else:
        print('Not a valid option')
