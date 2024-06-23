import xml.etree.ElementTree as ET
import csv
import re
import logging
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def parse_config_file(file_path):
    read_file   = None
    consulta_files = None
    resultado_file = None
    
    logging.info(f"Parsing configuration file: {file_path}")
    with open(file_path, 'r') as file:
        lines = file.readlines()
        
        for line in lines:
            key, value = line.strip().split('=', 1)
            key = key.strip()
            value = value.strip()
            
            if key == "LEIA":
                read_file = value.strip('"')
                logging.info(f"Input query file added: {value}")
            elif key == "CONSULTAS":
                consulta_files = value.strip('"')
                logging.info(f"Output query file added: {value}")
            elif key == "ESPERADOS":
                resultado_file = value.strip('"')
                logging.info(f"Expected Result file set to: {resultado_file}")    

    logging.info(f"Configuration file parsed successfully. Input Query file: {read_file}, Output Query files: {consulta_files},Expected Result file: {resultado_file}")
    return read_file, consulta_files, resultado_file

def clean_text(text):
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    cleaned_tokens = [re.sub(r'\W+|\d+', '', word).upper() for word in filtered_tokens if re.sub(r'\W+|\d+', '', word).upper() != ""]
    return cleaned_tokens





def process_query(query_elem):
    query_number = query_elem.find('QueryNumber').text.strip()
    query_text = query_elem.find('QueryText').text.strip()
    results = query_elem.find('Results').text.strip()

    cleaned_query = clean_text(query_text)

    records = []
    for record in query_elem.findall('.//Item'):
        score = record.get('score')
        item = record.text.strip()
        records.append((item, score))

    return query_number, cleaned_query, results, records

def calculate_score(vote_str):
    return sum(1 for digit in vote_str if digit != '0')

def build_query_file(config_file):
    xml_file, txt_file_name, expected_result_file = parse_config_file(config_file)
    logging.info(f"Started processing XML file: {xml_file}")

    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        with open(txt_file_name, 'w', newline='', encoding='utf-8') as txt_file, open(expected_result_file, 'w', newline='', encoding='utf-8') as result_file:
            writer = csv.writer(result_file, delimiter=',')
            
            for query in root.findall('.//QUERY'):
                query_number, cleaned_query, results, records = process_query(query)
                
                # Write to the query file
                txt_file.write(f'CONSULTA = ["{query_number}","{cleaned_query}"]\n')
                
                # Process and write to the results file
                sorted_records = sorted(records, key=lambda x: -calculate_score(x[1]))[:10]
                for item, score in sorted_records:
                    writer.writerow([query_number, item, score])
                
        logging.info(f"Query file generated: {txt_file_name}")
        logging.info(f"Results file generated: {expected_result_file}")

    except Exception as e:
        logging.error(f"Error processing XML file {xml_file}: {e}")

    logging.info("Finished processing.")





