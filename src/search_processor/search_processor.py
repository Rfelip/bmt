import xml.etree.ElementTree as ET
import csv
import re
import logging
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def clean_text(text):
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    cleaned_tokens = [re.sub(r'\W+|\d+', '', word).upper() for word in filtered_tokens if re.sub(r'\W+|\d+', '', word).upper() != ""]
    return cleaned_tokens

def process_query(query_elem):
    query_number = query_elem.find('QueryNumber').text.strip()
    query_text = query_elem.find('QueryText').text.strip()
    
    cleaned_query = clean_text(query_text)
    return query_number, cleaned_query

def build_query_file(xml_file, txt_file_name):
    logging.info(f"Started processing XML file: {xml_file}")
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        with open(txt_file_name, 'w', newline='', encoding='utf-8') as txt_file:
            for query in root.findall('.//QUERY'):
                query_number, cleaned_query = process_query(query)
                txt_file.write(f'CONSULTA = ["{query_number}","{cleaned_query}"]\n')

        logging.info(f"query file generated: {txt_file_name}")

    except Exception as e:
        logging.error(f"Error processing XML file {xml_file}: {e}")

    logging.info("Finished processing.")