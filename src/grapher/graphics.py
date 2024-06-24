import csv
import logging

def load_results(filename):
    logging.info(f"Loading results from {filename}...")
    results = {}
    with open(filename, 'r') as csvfile:
        for line in csvfile:
            line = line.strip()
            if line.startswith("QUERY_NUM"):
                continue
            
            parts = line.split(',', maxsplit=1)
            query_num = int(parts[0].strip('"'))
            
            tuples = []
            line = parts[1]
            for i in range(9):
                split_index = line.find("),")
                tuple_str = line[1:split_index]
                rank, doc_id, cos_similarity = tuple_str.split(',')
                tuples.append((rank, doc_id, cos_similarity))
                line = line[split_index+2:]

            rank, doc_id, cos_similarity =  line[1:-2].split(",")
            tuples.append((rank, doc_id, cos_similarity))
            results[query_num] = tuples
    logging.info(f"Finished loading results from {filename}.")
    return results


import csv

def load_expected_data(file_path):
    results_dict = {}
    logging.info(f"Loading expected results from {file_path}...")
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if len(row) >= 3:
                query_num = row[0].strip()
                doc_Id = row[1].strip()
                score = row[2].strip()

                score_points = 0

                for digit in score:
                    if digit != '0':
                        score_points += 1

                if query_num in results_dict:
                    results_dict[query_num].append((doc_Id, score_points))
                else:
                    results_dict[query_num] = [(doc_Id, score_points)]

    logging.info(f"Finished loading expected results from {file_path}. Ranking the expected results.")
    #Rankeando as tuplas...
    ranked_results = {}
    for query_num in results_dict:
        results_for_query = results_dict[query_num]
        results_for_query.sort(key=lambda x: x[1], reverse=True) 

        ranked_results[int(query_num)] = [(rank + 1, doc_Id, score_points) for rank, (doc_Id, score_points) in enumerate(results_for_query)]

    return ranked_results

