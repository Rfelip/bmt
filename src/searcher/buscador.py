import logging
import csv
import numpy as np


def load_vec_model(file_path):
    word_dict = {}
    document_ids = []
    
    logging.info(f"Loading vector model from file: {file_path}")
    with open(file_path, 'r', newline='') as csvfile:
        csvreader = csv.reader(csvfile)
        
        headers = next(csvreader)
        document_ids = headers[1:] 

        for row in csvreader:
            word = row[0]
            word_dict[word] = row[1:]
    
    logging.info(f"Vector model loaded successfully with {len(word_dict)} words and {len(document_ids)} documents")
    return word_dict, document_ids

def read_query_file(file_path):
    logging.info(f"Reading query file: {file_path}")
    consultas = []
    
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    for line in lines:
        if "CONSULTA =" in line:
            consulta_part = line.split('=')[1].strip()
            name_of_search, words_list_str = consulta_part.split(',', 1)
            
            name_of_search = name_of_search.replace("[", "").replace("]", "").replace("'", "").replace('"',"").strip()
            # Handle the case where the words list is not properly formatted
            words_list_str = words_list_str.replace("[", "").replace("]", "").replace("'", "").replace('"',"").strip()
            words_list = [word.strip() for word in words_list_str.split(',')]
            
            consultas.append((name_of_search, words_list))
    
    if not consultas:
        logging.warning(f"No query lines found in query file: {file_path}")
    
    logging.info(f"Query file '{file_path}' read successfully. Number of Queries: {len(consultas)}")
    return consultas


def search_word_vectors(query_file_path, word_dict, document_ids):
    queries = read_query_file(query_file_path)
    
    results = []
    
    for name_of_search, search_words in queries:
        
        vector_result = np.zeros(len(document_ids))
        for word in search_words:
            if word in word_dict:
                word_vector = np.array(word_dict[word], dtype=float)
                vector_result = vector_result + word_vector
        
        if np.linalg.norm(vector_result) == 0:
            logging.error(f"No word in query {name_of_search} found in model. words in query: {search_words}")
            continue

        vector_result = vector_result / np.linalg.norm(vector_result)
    
        doc_value_pairs = list(zip(document_ids, vector_result))
        
        doc_value_pairs.sort(key=lambda x: x[1], reverse=True)
        
        sorted_document_ids = [doc_id for doc_id, value in doc_value_pairs]
        
        results.append((name_of_search, sorted_document_ids, vector_result.copy()))
    
    logging.info(f"Results sorted for all queries in file '{query_file_path}'. Total queries: {len(results)}")
    return results

def parse_config_file(file_path):
    modelo_file = None
    consulta_files = []
    resultado_file = None
    
    logging.info(f"Parsing configuration file: {file_path}")
    with open(file_path, 'r') as file:
        lines = file.readlines()
        
        for line in lines:
            key, value = line.strip().split('=', 1)
            key = key.strip()
            value = value.strip()
            
            if key == "MODELO":
                if modelo_file is not None:
                    raise ValueError("Multiple models found.")
                modelo_file = value.strip('"')
                logging.info(f"Model file set to: {modelo_file}")
            elif key == "CONSULTAS":
                consulta_files.append(value.strip('"'))
                logging.info(f"Query file added: {value}")
            elif key == "RESULTADOS":
                if resultado_file is not None:
                    raise ValueError("Multiple results lines found.")
                resultado_file = value.strip('"')
                logging.info(f"Result file set to: {resultado_file}")
    
    if modelo_file is None:
        raise ValueError("Model line not found.")
    if not consulta_files:
        raise ValueError("Query line not found.")
    if resultado_file is None:
        raise ValueError("Results line not found.")
    
    logging.info(f"Configuration file parsed successfully. Model file: {modelo_file}, Query files: {consulta_files}, Result file: {resultado_file}")
    return modelo_file, consulta_files, resultado_file


def search_from_config(config_file_path, index):
    modelo_file, consulta_files, resultado_file = parse_config_file(config_file_path)

    word_dict, document_ids = load_vec_model(modelo_file)
    text_dict = create_document_word_dict(index)  # Used for document distance.

    with open(resultado_file, 'w') as result_file:
        result_file.write(f"QUERY_NUM, RESULTS (RANK, DOC_ID, COSINE_SIMILARITY)\n")
        for consulta_file in consulta_files:

            results = search_word_vectors(consulta_file, word_dict, document_ids)
            
            for name_of_search, sorted_document_ids, vector_result in results:
                result_file.write(f"{name_of_search}")
                for rank, doc_id in enumerate(sorted_document_ids[:10], start=1):
                    result_file.write(f",")
                    distance_result = document_distance(vector_result, doc_id, word_dict, text_dict)
                    result_file.write(f"({rank}, {int(doc_id)}, {distance_result})")
                result_file.write(";\n")
    
    logging.info(f"Operation completed. Results written to file: {resultado_file}")

def document_distance(vector, doc_id, word_dict, text_dict):
    vector_result = np.zeros(len(text_dict))

    if doc_id in text_dict:
        word_vector = text_dict[doc_id]
    else:
        logging.warning(f"Document ID '{doc_id}' not found in text_dict.")
        raise Exception

    for word in word_vector:
        word_vector = np.array(word_dict[word], dtype=float)
        vector_result += word_vector

    vector_result = vector_result/np.linalg.norm(vector_result)
    return np.dot(vector_result,vector)

def create_document_word_dict(index):
    document_words = {}

    for word, doc_ids in index.items():
        for doc_id in doc_ids:
            if doc_id in document_words:
                document_words[doc_id].append(word)
            else:
                document_words[doc_id] = [word]

    return document_words

