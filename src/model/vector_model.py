import csv
import math
import os
import logging
from datetime import datetime
from collections import defaultdict, Counter

def read_index_config(config_file):
    logging.info("Started reading index config file")
    index_file = ""
    tfidf_file = ""
    
    with open(config_file, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line.startswith("LEIA="):
                index_file = line.split('=', 1)[1].strip('"')
            elif line.startswith("ESCREVA="):
                tfidf_file = line.split('=', 1)[1].strip('"')
    
    logging.info("Finished reading index config file")
    return index_file, tfidf_file

def load_index(index_file):
    logging.info(f"Started loading index from {index_file}")
    index = {}
    
    with open(index_file, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        for row in reader:
            word = row[0]
            occurrences = eval(row[1])  # Convert string representation of list to list
            index[word] = occurrences
    
    logging.info(f"Finished loading index from {index_file}")
    return index

def create_document_word_count_dict(index):
    document_word_count = {}

    for word, doc_ids in index.items():
        for doc_id in doc_ids:
            if doc_id in document_word_count:
                document_word_count[doc_id] += 1
            else:
                document_word_count[doc_id] = 1
    return document_word_count

def calculate_tf_idf(index):

    document_word_count = create_document_word_count_dict(index)
    num_documents = len(document_word_count)

    # Step 1: Calculate TF (Term Frequency) for each document with adjusted formula
    tf = defaultdict(dict)
    for word, docs in index.items():
        doc_counts = Counter(docs)
        for doc_id, count in doc_counts.items():
            tf[word][doc_id] = 0.5 + 0.5 * (count / document_word_count[doc_id])

    # Step 2: Calculate IDF (Inverse Document Frequency)
    idf = {}
    for word, docs in index.items():
        num_docs_containing_word = len(set(docs))
        idf[word] = math.log(num_documents / num_docs_containing_word)

    # Step 3: Calculate TF-IDF
    tf_idf = defaultdict(dict)
    for word, docs in index.items():
        for doc_id in document_word_count.keys():
            if doc_id in tf[word]:
                tf_idf[word][doc_id] = tf[word][doc_id]* idf[word]
            else:
                tf_idf[word][doc_id] = 0.5 * idf[word]
    return tf_idf





def calculate_metric(index, metric):
    logging.info("Started calculating Metric for Analysis... (default is tf-idf)")
    matrix_term_document = metric(index)
    logging.info("Finished calculating metric.")
    return matrix_term_document

def write_metric_to_csv(tfidf_scores, tfidf_file):
    logging.info(f"Started writing METRIC scores to {tfidf_file}")
    with open(tfidf_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['WORD'] + list(tfidf_scores[next(iter(tfidf_scores))].keys()))  # Write header
        for record_num, scores in tfidf_scores.items():
            writer.writerow([record_num] + [scores[word] for word in scores])
    logging.info(f"Finished writing METRIC scores to {tfidf_file}")

def load_metric(tfidf_file):
    logging.info(f"Checking if METRIC file {tfidf_file} exists")
    tfidf_scores = {}
    
    if os.path.exists(tfidf_file):
        with open(tfidf_file, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader)  # Read header
            words = header[1:]  # Skip first column (RECORDNUM)
            
            for row in reader:
                record_num = row[0]
                scores = {words[i]: float(row[i + 1]) for i in range(len(words))}
                tfidf_scores[record_num] = scores
        
        logging.info(f"Loaded METRIC scores from {tfidf_file}")
    else:
        logging.info(f"METRIC file {tfidf_file} not found. Calculating METRIC scores.")
    
    return tfidf_scores

def build_vector_model(index_config_file, metric = calculate_tf_idf):
    start_time = datetime.now()
    logging.info("Vectorial model execution started")

    index_file, tfidf_file = read_index_config(index_config_file)
    index = load_index(index_file)
    tfidf_scores = load_metric(tfidf_file)
    
    if not tfidf_scores:
        tfidf_scores = calculate_metric(index, metric)
        # Step 5: Write METRIC scores to CSV
        write_metric_to_csv(tfidf_scores, tfidf_file)
    else:
        logging.info("Skipping METRIC calculation as results already exist.")
    
    end_time = datetime.now()
    logging.info("Vectorial model execution finished")
    logging.info(f"Total execution time: {end_time - start_time}")