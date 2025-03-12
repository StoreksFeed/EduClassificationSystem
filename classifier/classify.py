import re
import string
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from http.server import HTTPServer, BaseHTTPRequestHandler
from cassandra.cluster import Cluster
import uuid

def fprint(text, length=80):
    padding = (length - len(text) - 2) // 2
    print(f"{'-' * padding} {text} {'-' * padding}")


def clean_text(text):
    text = text.lower()
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def get_global_model(model_name):
    # check if there are global variables
    if model_name not in globals():
        # if not, create a global variable
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


def best_kmeans_n_clusters(text_e):
    # Determine the optimal number of clusters using Silhouette Score
    range_n_clusters = list(range(2, len(text_e) - 1))
    best_n_clusters = 2
    best_silhouette_score = -1
    for n_clusters in range_n_clusters:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(text_e)
        silhouette_avg = silhouette_score(text_e, cluster_labels)
        print(f'For n_clusters = {n_clusters}, the average score is : {silhouette_avg}')
        if silhouette_avg > best_silhouette_score:
            best_silhouette_score = silhouette_avg
            best_n_clusters = n_clusters
    print(f'Optimal number of clusters: {best_n_clusters}')
    return best_n_clusters

def classify():
    cluster = Cluster(['db'])
    session = cluster.connect('coursework')
    rows = session.execute('SELECT * FROM entry')
    rows = [(row.uuid, row.text) for row in rows]
    embeddings = np.array(list(get_bert_embeddings(clean_text(row[1])) for row in rows)).reshape((-1, 768))

    # Calculate cross text cosine similarity
    similarity_text = cosine_similarity(embeddings, embeddings)
    print(similarity_text)

    # Perform K-Means clustering with the optimal number of clusters
    fprint('K-Means Clustering')
    best_n_clusters = best_kmeans_n_clusters(embeddings)
    kmeans = KMeans(n_clusters=best_n_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)

    # Generate UUIDs for each label
    label_to_uuid = {label: str(uuid.uuid4()) for label in labels}

    # Group texts based on clustering labels
    grouped_texts = {}
    grouped_embeddings = {}
    for label, row_uuid, row_embedding in zip(labels, [row[0] for row in rows], embeddings):
        if label_to_uuid[label] not in grouped_texts:
            grouped_texts[label_to_uuid[label]] = []
            grouped_embeddings[label_to_uuid[label]] = []
        grouped_texts[label_to_uuid[label]].append(row_uuid)
        grouped_embeddings[label_to_uuid[label]].append(row_embedding)

    fprint('Grouped Texts')
    # Print grouped texts
    for uuid_label, group in grouped_texts.items(): # grouped is with uuids in labels
        print(f'Group {uuid_label}:')
        for row_uuid in group:
            print(f'  - {row_uuid}')
            session.execute(f"UPDATE entry SET group = {uuid_label} WHERE uuid = {row_uuid}")


    # Seems to not work properly. Yet.
    # common_vectors = {}
    # for uuid_label, embeddings in grouped_embeddings.items(): # grouped is with uuids in labels
    #     common_vectors[uuid_label] = np.mean(embeddings, axis=0).tolist()
    #     session.execute(f"INSERT INTO group (uuid, vector) VALUES (%s, %s)", (uuid_label, common_vectors[uuid_label]))

    # fprint('Common Vectors')
    # similarity_1 = cosine_similarity(embeddings, list(common_vectors.values()))
    # for text, sim in zip([row.text for row in rows], similarity_1):
    #     print(f'{text[:20]} -> {sim}')


class RequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/plain')
        self.end_headers()
        classify()
        self.wfile.write('Classification completed'.encode())

def run(server_class=HTTPServer, handler_class=RequestHandler, port=8010):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print(f'Starting httpd server on port {port}')
    httpd.serve_forever()

if __name__ == '__main__':
    run()
