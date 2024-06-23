import xml.etree.ElementTree as ET
import re
from collections import defaultdict
import csv
import logging
from datetime import datetime
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def read_config(config_file):
    ### Reads the config file.
    logging.info("Started reading config file")
    read_files = []
    write_file = ""
    
    with open(config_file, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line.startswith("LEIA="):
                read_files.append(line.split('=', 1)[1].strip('"'))
            elif line.startswith("ESCREVA="):
                write_file = line.split('=', 1)[1].strip('"')
    
    logging.info("Finished reading config file")
    return read_files, write_file

def parse_xml(file):
    logging.info(f"Started reading data from {file}")
    records = []
    tree = ET.parse(file)
    root = tree.getroot()
    
    for record in root.findall('.//RECORD'):
        record_num_element = record.find('RECORDNUM')
        if record_num_element is None or record_num_element.text is None:
            continue
        record_num = record_num_element.text.strip()
        
        abstract_element = record.find('ABSTRACT')
        abstract = abstract_element.text.strip() if abstract_element is not None and abstract_element.text is not None else None
        
        if abstract is None:
            # Look for the ABSTRACT in the EXTRACT element
            extract_element = record.find('EXTRACT')
            if extract_element is not None and extract_element.text is not None:
                abstract = extract_element.text.strip()
        
        if abstract is not None:
            records.append((record_num, abstract))
    
    logging.info(f"Finished reading data from {file}")
    return records



def clean_word(word):
    # Remove non-word characters and numbers, then convert to uppercase
    return re.sub(r'\W+|\d+', '', word).upper()


def index_words(records):
    logging.info("Started indexing words")
    index = defaultdict(list)
    stop_words = set(stopwords.words('english') + ["the"])
    for record_num, abstract in records:
        words = re.findall(r'\b\w+\b', abstract)
        for word in words:
            cleaned_word = clean_word(word)
            if cleaned_word.lower() in stop_words:
                continue
            else:
                if cleaned_word and len(cleaned_word) >= 3:  # Only add non-empty words
                    index[cleaned_word].append(record_num)
    
    logging.info("Finished indexing words")
    return index

def write_index_to_csv(index, output_file):
    logging.info(f"Started writing index to {output_file}")
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=';')
        for word, occurrences in index.items():
            writer.writerow([word.upper(), occurrences])
    logging.info(f"Finished writing index to {output_file}")

def index_files(config_file):
    logging.info("Script execution started")
    start_time = datetime.now()
    
    read_files, write_file = read_config(config_file)
    
    all_records = []
    for file in read_files:
        records = parse_xml(file)
        all_records.extend(records)
    
    index = index_words(all_records)
    write_index_to_csv(index, write_file)
    
    end_time = datetime.now()
    logging.info("Script execution finished")
    logging.info(f"Total files read: {len(read_files)}")
    logging.info(f"Total execution time: {end_time - start_time}")
    return index



