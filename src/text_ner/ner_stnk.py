import string
import numpy as np
from strsimpy.levenshtein import Levenshtein
from config import config as conf


def get_threshold(length):
    if length < 3:
        return 1
    elif length < 7:
        return 2
    else:
        return 3

def compare_strings(key, detected_text, levenshtein_score, levenshtein_closest_string, detection_top_left_corner, image_shape):
    threshold = get_threshold(len(key))
    score = Levenshtein().distance(key.lower().strip(), detected_text.lower().strip())
    if score < levenshtein_score and score < threshold and detection_top_left_corner[0] <= conf.KEY_FILED_REGION_THRESHOLD_X * image_shape[1]:
        return score, key, True
    return levenshtein_score, levenshtein_closest_string, False

def process_key_field_detections(detections, image_shape, key_fields):
    for detection in detections:
        levenshtein_closest_string = None
        levenshtein_score = np.inf
        for key in key_fields:
            levenshtein_score, levenshtein_closest_string, _ = compare_strings(key, detection['text'], levenshtein_score, levenshtein_closest_string, detection['top_left'], image_shape)
        detection['levenshtein_score'] = levenshtein_score
        detection['levenshtein_closest_string'] = levenshtein_closest_string
    return detections

def clean_key_field_detections(detections: list = None) -> list:
    key_detections = [i for i in detections if i['levenshtein_closest_string']]
    dist_matrix = np.zeros((len(key_detections), len(key_detections)), dtype='object')
    for i in range(len(key_detections)):
        for j in range(len(key_detections)):
            if j > i:
                x1, y1 = key_detections[j]['top_left']
                x2, y2 = key_detections[i]['top_left']
                y = -y2-(-y1)
                x = 0.0001 if x2-x1 == 0 else x2-x1
                distance = np.linalg.norm(np.array((x1, y1))-np.array((x2, y2)))
                slope = np.arctan(y/x) * 180 / np.pi
                dist_matrix[j][i] = (key_detections[j]['levenshtein_closest_string'], key_detections[i]['levenshtein_closest_string'], distance, slope)
    return key_detections, dist_matrix

def cluster_key_field_detections(key_detections, dist_matrix):
    final_detections = []
    clusterd_words = []
    rows, columns = dist_matrix.shape
    if (rows == 1 or columns == 1) or (rows == 0 or columns == 0):
        return key_detections
    for i in range(len(key_detections)):
        flag = True
        start_word_index = i
        end_word_index = None
        same_line_words = ''
        same_line_words_list = []
        for j in range(i+1, len(key_detections)):
            if abs(dist_matrix[j][i][3]) <= conf.ANGLE_THRESHOLD and dist_matrix[j][i][2] <= conf.DISTANCE_THRESHOLD:
                same_line_words_list.append(dist_matrix[j][i][0])
                if [key_detections[i]['levenshtein_closest_string']] + same_line_words_list in clusterd_words:
                    end_word_index = None
                    flag = False
                else:
                    end_word_index = j
                    same_line_words += f" {dist_matrix[j][i][0]}"
        clusterd_words.append(same_line_words_list)
        if end_word_index:
            cluster = {
                'text': key_detections[start_word_index]['text'] + f" {same_line_words}", 
                'top_left': key_detections[start_word_index]['top_left'], 
                'top_right': key_detections[end_word_index]['top_right'], 
                'bottom_right': key_detections[end_word_index]['bottom_right'], 
                'bottom_left': key_detections[start_word_index]['bottom_left'], 
                'width': key_detections[end_word_index]['top_right'][0] - key_detections[start_word_index]['top_left'][0],
                'height': key_detections[end_word_index]['bottom_right'][1] - key_detections[end_word_index]['top_right'][1],
                'levenshtein_score': 0, 
                'levenshtein_closest_string': key_detections[start_word_index]['levenshtein_closest_string'] + f" {same_line_words}"
            }
            final_detections.append(cluster)
        if flag and not end_word_index and [key_detections[start_word_index]['levenshtein_closest_string']] not in clusterd_words:
            final_detections.append(key_detections[start_word_index])
    return final_detections

def remove_unwanted_key_field_detections(key_detections, duplicate_tokens):
    res_key_detections = []
    for key in key_detections:
        for token in duplicate_tokens:
            score = Levenshtein().distance(token[0].lower().strip(), key['text'].lower().strip())
            if score >= token[1]:
                res_key_detections.append(key)
    return res_key_detections

def get_key_fields(detections, image_shape, key_fields, duplicate_tokens):
    key_detections, dist_matrix = clean_key_field_detections(process_key_field_detections(detections, image_shape, key_fields))
    key_fields_detections_clusterd = cluster_key_field_detections(key_detections, dist_matrix)
    key_fields_detections_clusterd = remove_unwanted_key_field_detections(key_fields_detections_clusterd, duplicate_tokens)
    return key_fields_detections_clusterd

def get_associations(key_field_detection, detections, key_fields):
    for detection in detections:
        x1, y1 = key_field_detection['top_left']
        x2, y2 = detection['top_left']
        y_1 = -y2-(-y1)
        x_1 = 0.0001 if x2-x1 == 0 else x2-x1
        slope1 = np.arctan(y_1/x_1) * 180 / np.pi
        if slope1 >= conf.ANGLE_THRESHOLD_TOLERANCE_MIN and slope1 < conf.ANGLE_THRESHOLD_TOLERANCE_MAX and detection['top_left'][1] >= key_field_detection['top_left'][1]-conf.FORM_PIXEL_TOLERANCE and detection['top_left'][0] >= key_field_detection['top_left'][0] and detection['levenshtein_closest_string'] not in key_fields:
            detection['mapping_with'] = key_field_detection['text']
            detection['association'] = key_field_detection['levenshtein_closest_string']
            detection['slope1'] = slope1
    return detections

def clean_associations(key_field_detection, detections, image_height):
    # Note: the first and the last item in key_field_detection are always the first and the last keywords respectively.
    cleaned_associations = []
    upper_height_threshold = key_field_detection[0]['top_left'][1] - conf.FORM_PIXEL_TOLERANCE if key_field_detection else 0
    lower_height_threshold = key_field_detection[-1]['top_left'][1] + conf.FORM_PIXEL_TOLERANCE if key_field_detection else image_height
    for detection in detections:
        if detection['top_left'][1] >= upper_height_threshold and detection['top_left'][1] <= lower_height_threshold and 'association' in detection.keys() and detection['text'] not in string.punctuation:
            cleaned_associations.append(detection)
    return cleaned_associations

def cluster_associations_detections(key_detections, cleaned_detections, image_width, form_threshold=0.5, key_threshold=None):
    clustered_associations, dict_key_word_associations = [], {}
    dict_key_word_associations = {i['levenshtein_closest_string']: [] for i in key_detections}
    dict_key_word_vertices = {i['levenshtein_closest_string']: {
        'top_left': i['top_left'],
        'top_right': i['top_right'],
        'bottom_right': i['bottom_right'],
        'bottom_left': i['bottom_left']
    } for i in key_detections}
    print(dict_key_word_associations)
    # create dynamic threshold dict from the one in config.
    true_keys = key_threshold.keys()
    dyncamic_key_threshold = {}
    for key in dict_key_word_associations.keys():
        for true_key in true_keys:
            if key == 'nomor  registrasi':
                print('Before   ->   ', key, '   ', true_key)
                print(true_key.lower().strip().replace('  ', ' ').replace('_', ' '))
                print(key.lower().replace('  ', ' ').strip())
                print(Levenshtein().distance(true_key.lower().strip().replace('  ', ' ').replace('_', ' '), key.lower().replace('  ', ' ').strip()))
                print(true_key in key or key in true_key)
                print(Levenshtein().distance(true_key.lower().strip().replace('  ', ' ').replace('_', ' '), key.lower().replace('  ', ' ').strip()) <= 4)
                print((true_key in key or key in true_key) and Levenshtein().distance(true_key.lower().strip().replace('  ', ' ').replace('_', ' '), key.lower().replace('  ', ' ').strip()) <= 4)
            if (true_key in key or key in true_key) and Levenshtein().distance(true_key.lower().strip().replace('  ', ' ').replace('_', ' '), key.lower().replace('  ', ' ').strip()) <= 4:
                print(key, '   ', true_key)
                dyncamic_key_threshold[key] = key_threshold[true_key]
                break
    print(dyncamic_key_threshold)
    for detection in cleaned_detections:
        dict_key_word_associations[detection['association']].append(detection)
    for key, value in dict_key_word_associations.items():
        if key == 'no.':
            print(key)
            print(value)
        '''
            Generate neighbours:
            eg: ABCD EFG HIJ             KLM UVW
                NOP QRST
            neighbours => (ABCD, EFG), (EFG, HIJ), (NOP, QRST), (KLM UVW)
        '''
        neighbours = {}
        for i, value_x in enumerate(value):
            dist = []
            for j, value_y in enumerate(value):
                if i == j:
                    continue
                if abs(value_y['top_left'][0] - value_x['top_right'][0]) <= conf.HORIZONTAL_DIST_BETWEEN_WORDS and abs(value_x['top_right'][1] - value_y['top_left'][1]) <= conf.VERTICAL_DIST_BETWEEN_WORDS:
                    dist.append([abs(value_y['top_left'][0] - value_x['top_right'][0]), j])
            if dist:       
                sorted(dist, key = lambda x: x[0])
                neighbours[i] = dist[0][1]
            else:
                neighbours[i] = None
        # Create reverse mapping for neighbours
        rev_neighbour_mapping = {conf.NONE_TOKEN: []}
        for k, v in neighbours.items():
            if v is None:
                rev_neighbour_mapping[conf.NONE_TOKEN].append(k)
            else:
                rev_neighbour_mapping[v] = k
        # Cluster neighbours
        clusters = []
        for end_token in rev_neighbour_mapping[conf.NONE_TOKEN]:
            cluster = [end_token]
            while cluster[-1] in rev_neighbour_mapping.keys():
                cluster.append(rev_neighbour_mapping[cluster[-1]])
            clusters.append(cluster)
        # Sort clusters based on their positions and get their true values
        true_clusters = []
        for cluster in clusters:
            true_cluster = [value[i] for i in cluster]
            true_cluster = sorted(true_cluster, key = lambda x: x['top_left'][0])
            true_clusters.append(true_cluster)
        true_clusters = sorted(true_clusters, key = lambda x: [x[0]['top_left'][1], x[0]['top_left'][0]])
        # Group items within a cluster
        merged_cluster_items = []
        for clusters in true_clusters:
            x_min, y_min, x_max, y_max, text = [], [], [], [], ''
            for item in clusters:
                text += f"{item['text'].strip()} "
                x_min.append(min(item['top_left'][0], item['bottom_left'][0]))
                y_min.append(min(item['top_left'][1], item['top_right'][1]))
                x_max.append(max(item['bottom_right'][0], item['top_right'][0]))
                y_max.append(max(item['bottom_right'][1], item['bottom_left'][1]))
            min_x = min(x_min)
            min_y = min(y_min)
            max_x = max(x_max)
            max_y = max(y_max)
            top_left, bottom_right, top_right, bottom_left = [min_x, min_y], [max_x, max_y], [max_x, min_y], [min_x, max_y]
            merged_cluster_items.append(
                {
                    "text": text.strip(),
                    "top_left": top_left,
                    "top_right": top_right,
                    "bottom_right": bottom_right,
                    "bottom_left": bottom_left,
                    'width': max_x - min_x,
                    'height': max_y - min_y,
                    'levenshtein_score': None, 
                    'levenshtein_closest_string': None,
                    'association': key,
                }
            )
        if key == 'no':
            remove_index = []
            print('*'*50)
            for ind, i in enumerate(merged_cluster_items):
                if not abs(dict_key_word_vertices[key]['top_left'][0] - i['top_left'][0]) <= key_threshold[key]:
                    remove_index.append(i)
            for i in remove_index:
                merged_cluster_items.remove(i)
            print('*'*50)
        # Get inference
        x_min, y_min, x_max, y_max, text = [], [], [], [], ''
        for clusters in merged_cluster_items:
            if clusters['top_left'][0] <= image_width * form_threshold:
                text += f"{clusters['text'].strip()} "
                x_min.append(min(clusters['top_left'][0], clusters['bottom_left'][0]))
                y_min.append(min(clusters['top_left'][1], clusters['top_right'][1]))
                x_max.append(max(clusters['bottom_right'][0], clusters['top_right'][0]))
                y_max.append(max(clusters['bottom_right'][1], clusters['bottom_left'][1]))
                clustered_associations.append(clusters)
        min_x = min(x_min) if x_min else None
        min_y = min(y_min) if y_min else None
        max_x = max(x_max) if x_max else None
        max_y = max(y_max) if y_max else None
        top_left, bottom_right, top_right, bottom_left = [min_x, min_y], [max_x, max_y], [max_x, min_y], [min_x, max_y]
        inference = {
            "text": text.strip(),
            "top_left": top_left,
            "top_right": top_right,
            "bottom_right": bottom_right,
            "bottom_left": bottom_left,
            'width': max_x - min_x if max_x and min_x else None,
            'height': max_y - min_y if max_y and min_y else None,
            'levenshtein_score': None, 
            'levenshtein_closest_string': None,
            'association': key,
            'cluster': True
        }
        if key == 'no.':
            print(inference)
        clustered_associations.append(inference)
    return clustered_associations

def get_associated_fields(key_detections, detections, image_height, image_width, key_fields, form_threshold=0.5, key_threshold=None):
    detections = sorted(detections, key=lambda x: [x['top_left'][1], x['top_left'][0]])
    key_detections = sorted(key_detections, key=lambda x: [x['top_left'][1], x['top_left'][0]])
    keywords = [i['text'] for i in key_detections]
    filtered_detections = [i for i in detections if i['text'] not in keywords]
    for _, key_field_detection in enumerate(key_detections):
        get_associations(key_field_detection, detections, key_fields)
    cleaned_detections = clean_associations(key_detections, filtered_detections, image_height)
    clustered_associations = cluster_associations_detections(key_detections, cleaned_detections, image_width, form_threshold, key_threshold)
    return clustered_associations