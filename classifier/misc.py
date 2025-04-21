import random
import sys
from json import load, dump
from uuid import uuid4, UUID

import numpy as np
from requests import get
from sklearn.metrics import accuracy_score, adjusted_rand_score, adjusted_mutual_info_score
from scipy.optimize import linear_sum_assignment
# Cassandra driver is not imported here, because it struggles to work with Windows

BENCH_RANGE = 2, 40


class Misc:
    """
    Miscellaneous utility functions for data processing and evaluation.
    """
    def __init__(Misc):
        pass

    @staticmethod
    def gen_uuids(input_file, output_file):
        """
        Generate UUIDs for objects in the input file and save the updated data to the output file.

        Args:
            input_file (str): Path to the input JSON file containing objects with 'content' and 'group' fields.
            output_file (str): Path to the output JSON file where the updated objects will be saved.
        """

        with open(input_file, encoding='utf-8') as f:
            objects = load(f)

        data = []

        for obj in objects:
            if obj['content']:
                data.append({
                    'uuid': str(uuid4()),
                    'text': obj['content'],
                    'label': obj['group']
                })

        with open(output_file, 'w', encoding='utf-8') as f:
            dump(data, f, ensure_ascii=False)

    @staticmethod
    def upload(input_file):
        """
        Upload data from a JSON file to the Cassandra database.

        Args:
            input_file (str): Path to the input JSON file containing objects with 'uuid', 'text', and 'group' fields.
        """

        from cassandra.cluster import Cluster

        cluster = Cluster(['db1', 'db2'])
        session = cluster.connect('educational_materials')

        session.execute('TRUNCATE entry')
        session.execute('TRUNCATE group')
        with open(input_file, encoding='utf-8') as f:
            data = load(f)
            for entry in data:
                session.execute(
                    f"INSERT INTO entry (uuid, text, status, group) VALUES (%s, %s, %s, %s)",
                    (UUID(entry['uuid']), entry['text'], 0, UUID(int=0))
                )

        cluster.shutdown()

    @staticmethod
    def download(output_file):
        """
        Download data from the Cassandra database and save it to a JSON file.

        Args:
            output_file (str): Path to the output JSON file where the data will be saved.
        """

        from cassandra.cluster import Cluster

        cluster = Cluster(['db1', 'db2'])
        session = cluster.connect('educational_materials')

        entries = session.execute('SELECT * FROM entry')
        data = []
        for entry in entries:
            data.append({
                'uuid': str(entry.uuid),
                'group': str(entry.group)
            })

        cluster.shutdown()

        with open(output_file, 'w', encoding='utf-8') as f:
            dump(data, f, ensure_ascii=False)

    @staticmethod
    def uuid_to_labels(input_file, output_file):
        """
        Convert UUIDs to labels based on clustering information in the input file.

        Args:
            input_file (str): Path to the input JSON file containing clustering information.
            output_file (str): Path to the output JSON file where the UUID-to-label mapping will be saved.
        """

        with open(input_file, encoding='utf-8') as f:
            objects = load(f)

        data = []
        if 'documents' in objects:  # Carrot file
            for i in range(len(objects['documents'])):
                labels = [cluster for cluster in range(len(objects['clusters']))
                          if i in objects['clusters'][cluster]['documents']]

                data.append({
                    'uuid': objects['documents'][i]['uuid'],
                    'label': labels
                })
        else:  # Our file
            labels = list(set(object['group'] for object in objects))

            for object in objects:
                data.append({
                    'uuid': object['uuid'],
                    'label': [labels.index(object['group'])]
                })

        with open(output_file, 'w', encoding='utf-8') as f:
            dump(data, f, ensure_ascii=False)

    @staticmethod
    def book_to_labels(input_file, output_file_h1, output_file_h2):
        """
        Convert hierarchical labels from the input file into two separate label files.

        Args:
            input_file (str): Path to the input JSON file containing objects with hierarchical labels.
            output_file_h1 (str): Path to the output JSON file for first-level labels.
            output_file_h2 (str): Path to the output JSON file for second-level labels.
        """

        with open(input_file, encoding='utf-8') as f:
            objects = load(f)

        h1_labels = set()
        h2_labels = set()
        for object in objects:
            label_parts = object['label'].split(' | ')
            h1_labels.add(label_parts[0])
            if len(label_parts) > 1:
                h2_labels.add(label_parts[0] + ' | ' + label_parts[1])
            else:
                h2_labels.add(object['label'])
        h1_labels = list(h1_labels)
        h2_labels = list(h2_labels)

        data = []
        for object in objects:
            data.append({
                'uuid': object['uuid'],
                'label': [h1_labels.index(object['label'].split(' | ')[0])]
            })

        with open(output_file_h1, 'w', encoding='utf-8') as f:
            dump(data, f, ensure_ascii=False)

        data = []
        for object in objects:
            label_parts = object['label'].split(' | ')
            if len(label_parts) > 1:
                label = label_parts[0] + ' | ' + label_parts[1]
            else:
                label = object['label']
            data.append({
                'uuid': object['uuid'],
                'label': [h2_labels.index(label)]
            })

        with open(output_file_h2, 'w', encoding='utf-8') as f:
            dump(data, f, ensure_ascii=False)

    @staticmethod
    def gen_range_labels(input_file, output_file, count):
        """
        Generate labels a range of labels (from 0 to count-1) for each object in the input file.

        Args:
            input_file (str): Path to the input JSON file containing objects with 'uuid' fields.
            output_file (str): Path to the output JSON file where the objects with generated labels will be saved.
            count (int): Number of labels to generate for each object.
        """

        with open(input_file, encoding='utf-8') as f:
            objects = load(f)

        data = []
        for object in objects:
            data.append({
                'uuid': object['uuid'],
                'label': [i for i in range(count)]
            })

        with open(output_file, 'w', encoding='utf-8') as f:
            dump(data, f, ensure_ascii=False)

    @staticmethod
    def clustering_accuracy(labels_true, labels_pred):
        """
        Compute clustering accuracy (ACC) by finding the best one-to-one mapping
        between predicted and true labels using the Hungarian algorithm.

        Args:
            labels_true (list or np.ndarray): Ground-truth labels.
            labels_pred (list or np.ndarray): Predicted labels.

        Returns:
            float: Clustering accuracy (ACC).
        """

        # Get unique labels from true and predicted labels
        unique_labels_true = list(set(labels_true))
        unique_labels_pred = list(set(labels_pred))

        # Initialize the confusion matrix
        confusion_matrix = np.zeros(
            (len(unique_labels_true), len(unique_labels_pred))
        )

        # Populate the confusion matrix
        for true, pred in zip(labels_true, labels_pred):
            confusion_matrix[unique_labels_true.index(
                true)][unique_labels_pred.index(pred)] += 1

        # Transpose the confusion matrix
        confusion_matrix = confusion_matrix.T
        
        # Use the Hungarian algorithm to find the optimal label mapping
        row_ind, col_ind = linear_sum_assignment(-confusion_matrix)
        
        # Create a mapping from predicted labels to true labels
        label_mapping = {unique_labels_pred[row]: unique_labels_true[col]
                         for row, col in zip(row_ind, col_ind)}
        
        # Remap predicted labels to match true labels
        remapped_labels_pred = np.array([
            label_mapping[label] if label in label_mapping else -1  # Handle unmapped labels
            for label in labels_pred
        ])
        
        # Compute and return the accuracy score
        return accuracy_score(labels_true, remapped_labels_pred)

    @staticmethod
    def get_metrics(true_labels_file, predicted_labels_file, bootstrap=1, to_console=True):
        """
        Compute evaluation metrics (ACC, NMI, ARI) between true and predicted labels.

        Args:
            true_labels_file (str): Path to the JSON file containing ground-truth labels.
            predicted_labels_file (str): Path to the JSON file containing predicted labels.
            bootstrap (int): Number of bootstrap iterations for metric computation.
            to_console (bool): Whether to print the results to the console.

        Returns:
            tuple: A tuple containing:
                - acc (tuple): Mean and standard deviation of clustering accuracy (ACC).
                - nmi (tuple): Mean and standard deviation of normalized mutual information (NMI).
                - ari (tuple): Mean and standard deviation of adjusted Rand index (ARI).
        """

        with open(true_labels_file, encoding='utf-8') as f:
            true_labels = load(f)
            true_labels = sorted(true_labels, key=lambda x: x['uuid'])

        with open(predicted_labels_file, encoding='utf-8') as f:
            predicted_labels = load(f)
            predicted_labels = sorted(
                predicted_labels, key=lambda x: x['uuid'])

        random.seed(69)

        acc, ami, ari = [], [], []

        for _ in range(bootstrap):
            i_true_labels = [random.choice(i['label']) for i in true_labels]
            i_predicted_labels = [random.choice(
                i['label']) for i in predicted_labels]

            acc.append(Misc.clustering_accuracy(
                i_true_labels, i_predicted_labels))
            ami.append(adjusted_mutual_info_score(
                i_true_labels, i_predicted_labels))
            ari.append(adjusted_rand_score(
                i_true_labels, i_predicted_labels))

        acc = (np.mean(acc), np.std(acc))
        ami = (np.mean(ami), np.std(ami))
        ari = (np.mean(ari), np.std(ari))

        if to_console:
            print(f'{true_labels_file} vs {predicted_labels_file}')
            print(f'ACC: {acc[0]:.3f} ± {acc[1]:.3f}, \t', end='')
            print(f'AMI: {ami[0]:.3f} ± {ami[1]:.3f}, \t', end='')
            print(f'ARI: {ari[0]:.3f} ± {ari[1]:.3f}, \t', end='')
            print()

        return acc, ami, ari

    @staticmethod
    def bench_random(input_file, output_file):
        """
        Benchmark clustering performance using randomly generated labels.

        Args:
            input_file (str): Path to the input JSON file containing objects with hierarchical labels.
            output_file (str): Path to the output file where the metrics will be saved.
        """

        Misc.book_to_labels(input_file, input_file + '_h1', input_file + '_h2')

        start = BENCH_RANGE[0]
        stop = BENCH_RANGE[1]

        # range, h1/h2, acc/ami/ari, mean/std
        metrics = np.zeros((stop - start + 1, 2, 3, 2))

        for i in range(start, stop + 1):
            print('Clusters:', i)
            Misc.gen_range_labels(input_file, f'{input_file}_{i}', i)
            metrics[i - start,
                    0] = Misc.get_metrics(f'{input_file}_h1', f'{input_file}_{i}', i**2)
            metrics[i - start,
                    1] = Misc.get_metrics(f'{input_file}_h2', f'{input_file}_{i}', i**2)

        np.save(output_file, metrics)

    @staticmethod
    def bench_system(input_file, output_file):
        """
        Benchmark clustering performance using the system's clustering algorithm.

        Args:
            input_file (str): Path to the input JSON file containing objects with hierarchical labels.
            output_file (str): Path to the output file where the metrics will be saved.
        """

        Misc.book_to_labels(input_file, input_file + '_h1', input_file + '_h2')
        Misc.upload(input_file)

        start = BENCH_RANGE[0]
        stop = BENCH_RANGE[1]

        # range, h1/h2, acc/ami/ari, mean/std
        metrics = np.zeros((stop - start + 1, 2, 3, 2))

        for i in range(start, stop + 1):
            print('Clusters:', i)

            get(f'http://localhost:8001/clustering/{i}')
            
            Misc.download(f'{input_file}_{i}')

            Misc.uuid_to_labels(f'{input_file}_{i}', f'{input_file}_{i}')
            metrics[i - start,
                    0] = Misc.get_metrics(f'{input_file}_h1', f'{input_file}_{i}')
            metrics[i - start,
                    1] = Misc.get_metrics(f'{input_file}_h2', f'{input_file}_{i}')

        np.save(output_file, metrics)
        

    @staticmethod
    def bench_carrot(input_file_prefix, output_file_prefix):
        """
        Benchmark clustering performance using Carrot2's clustering algorithms (STC and Lingo).

        Args:
            input_file_prefix (str): Prefix for the input JSON files containing hierarchical labels.
            output_file_prefix (str): Prefix for the output files where the metrics will be saved.
        """

        Misc.book_to_labels(input_file_prefix, input_file_prefix +
                            '_h1', input_file_prefix + '_h2')

        start = BENCH_RANGE[0]
        stop = BENCH_RANGE[1]

        # range, h1/h2, acc/ami/ari, mean/std
        stc_metrics = np.zeros((stop - start + 1, 2, 3, 2))
        lingo_metrics = np.zeros((stop - start + 1, 2, 3, 2))

        for i in range(start, stop + 1):
            print('Clusters:', i)

            stc_metrics[i - start, 0] = Misc.get_metrics(
                f'{input_file_prefix}_h1', f'{input_file_prefix}_stc_{i}', i**2)
            stc_metrics[i - start, 1] = Misc.get_metrics(
                f'{input_file_prefix}_h2', f'{input_file_prefix}_stc_{i}', i**2)

            lingo_metrics[i - start, 0] = Misc.get_metrics(
                f'{input_file_prefix}_h1', f'{input_file_prefix}_lingo_{i}', i**2)
            lingo_metrics[i - start, 1] = Misc.get_metrics(
                f'{input_file_prefix}_h2', f'{input_file_prefix}_lingo_{i}', i**2)

        np.save(f'{output_file_prefix}_stc.npy', stc_metrics)
        np.save(f'{output_file_prefix}_lingo.npy', lingo_metrics)


COMMANDS = {attr: getattr(Misc, attr)
            for attr in dir(Misc) if not attr.startswith('__')}

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python misc.py <option> <args>')
        print('Options:')
        for command in COMMANDS:
            print(f' - {command}')
    else:
        command = sys.argv[1]
        if command in COMMANDS:
            if sys.argv[2] == 'help':
                print(f'Usage: python misc.py {command} <args>')
                print(COMMANDS[command].__doc__)
            else:
                COMMANDS[command](*sys.argv[2:])
        else:
            print('Not a valid option')
