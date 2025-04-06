from collections import defaultdict
from http.server import HTTPServer, BaseHTTPRequestHandler
from json import dumps
import re
import sys
import uuid

from cassandra.cluster import Cluster
import numpy as np
import torch

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

from download import get_model, get_morph


def pad_pring(text, length=80):
    """
    Print a text with a specified length and padding.

    Args:
        text (str): Text to be printed.
        length (int): Length of the output line.
    """

    padding = (length - len(text) - 2) // 2
    print(f"{'-' * padding} {text} {'-' * padding}")


def clean_text(text):
    """
    Clean a given text by removing extra spaces, unwanted characters, stopwords,
    and applying lemmatization for Russian text.

    Args:
        text (str): Input text to be cleaned.

    Returns:
        str: Cleaned and preprocessed text.
    """

    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[^\w\s.,!?()_[]\-\u2013\u2014]', '', text)

    words = text.split()

    russian_stopwords, morph = get_morph()

    cleaned_words = [
        morph.parse(word)[0].normal_form
        for word in words
        if word.lower() not in russian_stopwords
    ]

    return ' '.join(cleaned_words)


def get_bert_embeddings(texts, model_name):
    """
    Get BERT embeddings for a given list of texts.

    Args:
        texts (list): List of texts to be embedded.
        model_name (str): Name of the BERT model to be used.

    Returns:
        np.ndarray: Array of BERT embeddings.
    """

    tokenizer, model = get_model(model_name)
    inputs = tokenizer(texts, return_tensors='pt',
                       padding=True, truncation=True)

    with torch.no_grad():
        outputs = model(**inputs)

    return torch.mean(outputs.last_hidden_state, dim=1).detach().numpy()


def best_clustering(embeddings):
    """
    Perform K-Means clustering on a given set of embeddings and return labels with the best silhouette score.

    Args:
        embeddings (np.ndarray): Array of embeddings to be clustered.

    Returns:
        np.ndarray: Array of cluster labels.
    """

    array = np.array([])

    for n_clusters in range(3, len(embeddings) // 2):
        labels = AgglomerativeClustering(
            n_clusters=n_clusters).fit_predict(embeddings)
        silhouette = silhouette_score(embeddings, labels)
        array = np.append(array, silhouette)
        print(f'For n_clusters = {n_clusters}, the score is : {silhouette}')

    best_n_clusters = np.argmax(array) + 3
    print(f'Best number of clusters: {best_n_clusters}')

    labels = AgglomerativeClustering(
        n_clusters=best_n_clusters).fit_predict(embeddings)

    return labels


def clustering(cluster_count=None):
    """"
    Perform clustering on the entries in the database.
    """

    cluster = Cluster(['db0', 'db1'])
    session = cluster.connect('educational_materials')
    entries = [{
        'uuid': row.uuid,
        'text': row.text
    } for row in
        session.execute('SELECT * FROM entry')
    ]
    embeddings = get_bert_embeddings(list(
        clean_text(entry['text']) for entry in entries
    ), 'DeepPavlov/rubert-base-cased').reshape((-1, 768))

    pad_pring('K-Means Clustering')
    if cluster_count:
        labels = AgglomerativeClustering(
            n_clusters=cluster_count).fit_predict(embeddings)
    else:
        labels = best_clustering(embeddings)

    label_to_uuid = {label: uuid.uuid4() for label in labels}
    uuid_labels = [label_to_uuid[label] for label in labels]

    grouped_texts = defaultdict(list)
    grouped_embeddings = defaultdict(list)
    for uuid_label, entry, row_embedding in zip(uuid_labels, entries, embeddings):
        grouped_texts[uuid_label].append(entry['uuid'])
        grouped_embeddings[uuid_label].append(row_embedding)

    pad_pring('Grouped Texts')
    session.execute('TRUNCATE group')
    for uuid_label, group in grouped_texts.items():
        print(f'Group {uuid_label}:')
        for entry_uuid in group:
            print(f'  - {entry_uuid}')
            session.execute(
                f"UPDATE entry SET group = {uuid_label}, status = 1 WHERE uuid = {entry_uuid}")

    common_vectors = {}
    for uuid_label, embeddings in grouped_embeddings.items():
        common_vectors[uuid_label] = np.mean(embeddings, axis=0).tolist()
        session.execute(f"INSERT INTO group (uuid, count, vector) VALUES (%s, %s, %s)",
                        (uuid_label, len(embeddings), common_vectors[uuid_label]))

    cluster.shutdown()

    return len(embeddings), len(grouped_texts)


def classification(entry_uuid):
    """
    Perform classification on a single entry in the database."

    Args:
        entry_uuid (str): UUID of the entry to be classified.
    """

    cluster = Cluster(['db0', 'db1'])
    session = cluster.connect('educational_materials')
    text = session.execute(
        f"SELECT text FROM entry WHERE uuid = {entry_uuid}"
    ).one().text

    embedding = get_bert_embeddings(list(
        clean_text(text)
    ), 'DeepPavlov/rubert-base-cased').reshape((-1, 768))

    groups = [
        {
            'uuid': group.uuid,
            'count': group.count,
            'vector': group.vector
        } for group in
        session.execute('SELECT * FROM group')
    ]
    similarity = cosine_similarity(
        embedding, [np.array(group['vector']) for group in groups]
    )

    best_group = None
    best_similarity = 0.0

    for sim, group in zip(similarity[0], groups):
        if sim >= 0.8 and sim > best_similarity:
            best_group = group
            best_similarity = sim

    if best_group:
        print(
            f'{entry_uuid} -> Best Group {best_group["uuid"]} with similarity {best_similarity}')
        session.execute(
            f"UPDATE entry SET group = {best_group['uuid']}, status = 2 WHERE uuid = {entry_uuid}"
        )
        session.execute(
            f"UPDATE group SET count = {best_group['count'] + 1} WHERE uuid = {best_group['uuid']}"
        )
        new_vector = (embedding[0] + best_group['count'] *
                      np.array(best_group['vector'])) / (best_group['count'] + 1)
        session.execute(
            f"UPDATE group SET vector = {new_vector.tolist()} WHERE uuid = {best_group['uuid']}"
        )
    else:
        print(f'{entry_uuid} -> No group found')
        new_group_uuid = uuid.uuid4()
        session.execute(
            f"INSERT INTO group (uuid, count, vector) VALUES (%s, %s, %s)",
            (new_group_uuid, 1, embedding[0].tolist())
        )
        session.execute(
            f"UPDATE entry SET group = {new_group_uuid}, status = 2 WHERE uuid = {entry_uuid}"
        )

    cluster.shutdown()


class RequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith('/healthcheck'):
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()

            message = {"status": "OK"}

            self.wfile.write(dumps(message).encode())

        elif self.path.startswith('/clustering/'):
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()

            try:
                cluster_count = int(self.path.split('/')[-1])
            except:
                cluster_count = None

            try:
                entries, groups = clustering(cluster_count)
                message = {"status": "OK",
                           "entries": entries, "groups": groups}
            except Exception as e:
                message = {"status": "ERROR", "error": str(e)}

            self.wfile.write(dumps(message).encode())

        elif self.path.startswith('/classification/'):
            uuid = self.path.split('/')[-1]
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()

            try:
                classification(uuid)
                message = {"status": "OK", "uuid": uuid}
            except Exception as e:
                message = {"status": "ERROR", "error": str(e)}

            self.wfile.write(dumps(message).encode())

        else:
            self.send_response(404)
            self.send_header('Content-type', 'application/json')
            self.end_headers()

            message = {"status": "ERROR", "error": "Not found"}

            self.wfile.write(dumps(message).encode())


def runserver(port):
    httpd = HTTPServer(('', port), RequestHandler)
    print(f'Starting httpd server on port {port}')
    httpd.serve_forever()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python classify.py <option> <args>')
    elif sys.argv[1] == 'runserver':
        runserver(int(sys.argv[2]))
    else:
        print('Not a valid option')
