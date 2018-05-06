from __future__ import division
import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
import pymongo
import datetime
import sys
import logging
import math
import os
import json
import operator
import requests

logging.basicConfig(level=logging.INFO, filename="log.txt",
                    format="%(asctime)s: %(levelname)s:  %(message)s")
console = logging.StreamHandler()
console.setLevel(logging.INFO)
# set a format which is simpler for console use
formatter = logging.Formatter('%(asctime)s: %(levelname)s:  %(message)s')
# tell the handler to use this format
console.setFormatter(formatter)
# add the handler to the root logger
logging.getLogger('').addHandler(console)


submissions_directory = '../JSON/users_submissions/sample/'
users = {}
problems = {}
similarity = {}
predictions = {}
all_problems = {}
error_obj = {}
all_levels = {
        'A2': 1,
        'B2': 2,
        'C2': 3,
        'A1': 4,
        'D2': 5,
        'B1': 6,
        'E2': 7,
        'C1': 8,
        'D1': 9,
        'E1': 10,
        'F2': 11,
        'F1': 12
    }
# Mongo database instance
db = None


def init_db():
    try:
        global db
        if db is not None:
            return
        else:
            init_connection()
    except UnboundLocalError:
        init_connection()


def init_connection():
    # in mongo DB
    uri = 'mongodb://mohabamroo:ghostrider1@ds241699.mlab.com:41699/bachelor'
    client = pymongo.MongoClient(uri,
                                 connectTimeoutMS=30000,
                                 socketTimeoutMS=None,
                                 socketKeepAlive=True)
    global db
    db = client.get_database()
    logging.info("connected to MongoDB")


def get_verdict(verdict):
    if(verdict == "OK"):
        return {"key": "OK", "score": 2}
    else:
        return {"key": "W", "score": 1}
    return {
        'WRONG_ANSWER': {"key": 'W', "score": 0},
        'OK': {"key": 'OK', "score": 5},
        'COMPILATION_ERROR': {"key": 'C', "score": 2},
        'RUNTIME_ERROR': {"key": 'R', "score": 2},
        'MEMORY_LIMIT_EXCEEDED': {"key": 'M', "score": 3},
        'TIME_LIMIT_EXCEEDED': {"key": 'T', "score": 3},
        'CHALLENGED': {"key": 'CH', "score": 3},
        'PARTIAL': {"key": 'P', "score": 4},
        'SKIPPED': {"key": 'S', "score": 1},
        'PRESENTATION_ERROR': {"key": "P", "score": 2}
    }[verdict]


# old function
def get_level_from_score(score):
    # rounding to nearst level
    return chr(int(round(score)) + 64)

# old function
def level_score(level):
    # ASCII of A is 65
    return ord(level) - 64

def score_to_level(score):
    # rounding to nearst level
    score = round(score)
    for key in all_levels:
        if all_levels[key] == score:
            return key
    logging.error("Did not find level for score: %s", str(score))
    return 'A1'


def level_to_score(level):
    try:
        return all_levels[level]
    except KeyError:
        logging.error("Uknown level: %s", level)
        return 1

def load_all_problems(mongo=1):
    # loads problem information for each problem
    if mongo == 1:
        init_db()
        problems_docs = db['problems'].find()
        for doc in problems_docs:
            all_problems[doc['id']] = doc
    else:
        file = open('recommender/JSON/problems.json', 'r')
        for problem_line in list(file):
            problem = json.loads(problem_line)
            all_problems[problem['id']] = problem


def save_problems_in_DB():
    init_db()
    file = open('JSON/problems.json', 'r')
    for problem_line in list(file):
        problem = json.loads(problem_line)
        db['problems'].insert(problem, check_keys=False)


def save_submissions():
    # saving summary/stats about each submission in collection submissions
    init_db()
    logging.info('Started saving user submissions')
    submissions_collection = db['submissions']
    for user in users:
        entry = {'user': user, 'problems': users[user]['problems']}
        submissions_collection.update(
            {'user': user}, entry, upsert=True, check_keys=False)
    logging.info('Finished saving user submissions')


def process_user(username, mongo_user=None):
    # handles proccssing users from eithr JSON files or Mongo DB
    # aggregates a final verdict for each problem
    users[username]['problems'] = {}
    if mongo_user != None:
        submissions_list = mongo_user['problems']
    else:
        file = open(submissions_directory + username + '.json', 'r')
        submissions_list = list(file)
    for submission_line in submissions_list:
        if mongo_user != None:
            submission = submission_line
        else:
            submission = json.loads(submission_line)

        link_splitted = submission['problem_link'].split('/')
        problem_id = link_splitted[2] + '' + link_splitted[4]
        verdict_object = get_verdict(submission['verdict'])
        score = verdict_object['score']
        if users[username]['problems'].get(problem_id, None) == None:
            users[username]['problems'][problem_id] = {
                'count': 1,
                verdict_object['key']: 1,
                'score': score,
                'title': submission['title']
            }
        else:
            users[username]['problems'][problem_id]['count'] += 1
            users[username]['problems'][problem_id][verdict_object['key']
                                                    ] = users[username]['problems'][problem_id].get(verdict_object['key'], 0) + 1
            users[username]['problems'][problem_id]['score'] += score


def process_files():
    # fetches users and their submissions from the saved JSON files
    count = 0
    for filename in os.listdir(submissions_directory):
        if filename.endswith(".json"):
            count += 1
            username = filename.split('.json')[0]
            # if count < 100 or username == "Med0b1011":
            users[username] = {}
            process_user(username)


def fetch_users_from_DB(limit=100):
    # fetches users and their submissions from Mongo DB
    # similar to process files
    init_db()
    logging.info("Fetching crawled users' submissions from Database")
    users_submissions = db['all_submissions'].find(limit=limit)
    for user in users_submissions:
        users[user['user']] = {}
        process_user(user['user'], user)
    logging.info(
        "Initialized users array which holds stats (summary)a bout each problem for each user")


def fetch_summarizied_submissions(limit=100):
    # fetches users and their submissions from Mongo DB
    # similar to process files
    init_db()
    logging.info("Fetching mini submissions from Database")
    users_submissions = db['submissions'].find()
    logging.info("fetched: %s", str(users_submissions.count()))
    for user in users_submissions:
        users[user['user']] = user
    logging.info(
        "Initialized users array which holds stats (summary)a bout each problem for each user")


def fetch_new_user_from_DB(username):
    # fetches target user and his submissions from Mongo DB
    init_db()
    logging.info("Fetching target user's submissions from Database")
    user_submissions = db['all_submissions'].find_one({'user': username})
    users[user_submissions['user']] = {}
    process_user(user_submissions['user'], user_submissions)
    logging.info("Pushed target user in users array")


def init_item_matrix():
    logging.info(
        "Initializing inverse relation between problems and users, who sloved each problem by the problem ID")
    for user in users:
        total_avg = 0
        users[user]['tags'] = {}
        for problem in users[user]['problems']:
            # avg score of all submissions for this user for this problem
            problem_avg = users[user]['problems'][problem]['score'] / \
                users[user]['problems'][problem]['count']
            users[user]['problems'][problem]['avg'] = problem_avg
            problem_obj = extract_problem_info(problem)
            problem_tags = problem_obj['tags']
            problem_level = problem_obj['level'] + str(problem_obj['div'])
            users[user]['problems'][problem]['avg'] = problem_avg

            # adds stats about each tag the user solved
            for tag in problem_tags:
                if users[user]['tags'].get(tag) == None:
                    users[user]['tags'][tag] = {}
                if users[user]['tags'][tag].get(problem_level) == None:
                    users[user]['tags'][tag][problem_level] = 1
                else:
                    users[user]['tags'][tag][problem_level] += 1

            # adds the user id to that dict of this problem, the core of the inverse relation
            if(problems.get(problem) == None):
                problems[problem] = {'users': {}}
            problems[problem]['users'][user] = problem_avg
            total_avg += problem_avg

        # avg tags per user
        for tag in users[user]['tags']:
            total_problems = 0
            for level in users[user]['tags'][tag]:
                total_problems += users[user]['tags'][tag][level]
            for level in users[user]['tags'][tag]:
                users[user]['tags'][tag][level] = users[user]['tags'][tag][level] / total_problems

        # computes a score for user
        try:
            total_avg = total_avg/len(users[user]['problems'].keys())
        except ZeroDivisionError:
            total_avg = 0
        users[user]['total_avg'] = total_avg
    maxx = 0
    max_p = None

    for pz in problems:
        if(len(problems[pz]['users']) > maxx):
            maxx = len(problems[pz]['users'])
            max_p = pz
    logging.info("Max problem: " + str(max_p) + " = " + str(maxx))

    # item_user_matrix = {}
    # for prob in problems:
    #     for user in users:
    #         item_user_matrix[problem][user] = var_rating
    logging.info(
        "Finished averaging users' scores and inverse relation initialization")


def user_user_sim(user, other_user, matrix):
    sum = 0
    sum_u = 0
    sum_v = 0
    try:
        # if similarity is already computed
        return matrix[other_user][user]
    except:
        if(user == other_user):
            return 1
    for problem in users[user]['problems']:
        for problem_2 in users[other_user]['problems']:
            if(problem == problem_2):
                r_u_i = users[user]['problems'][problem]['avg'] - \
                    users[user]['total_avg']
                r_v_i = users[other_user]['problems'][problem_2]['avg'] - \
                    users[other_user]['total_avg']
                sum += (r_u_i * r_v_i)
                sum_u += pow(r_u_i, 2)
                sum_v += pow(r_v_i, 2)
    try:
        sim = sum / (math.sqrt(sum_u) * math.sqrt(sum_v))
        sim = round(sim, 4)
    except ZeroDivisionError:
        return 0
    return sim


def compute_user_sim():
    # computes similarities between all users
    logging.info('Started computing similarity matrix')
    for user in users:
        similarity[user] = {}
        for other_user in users:
            similarity[user][other_user] = user_user_sim(
                user, other_user, similarity)
    logging.info('finished computing similarity matrix')
    logging.info('Started saving similarity matrix')
    init_db()
    new_matrix = {'created': datetime.datetime.utcnow(
    ), 'matrix': similarity, 'users_length': len(users)}
    db['similarity_matrix'].insert(new_matrix, check_keys=False)
    logging.info('Finished saving similarity matrix')


# computes predicted rating/output for user U for an item I
def compute_predictions(user, problem, matrix=None):
    score_sim_sum = 0
    sim_sum = 0
    try:
        similarity[user]
    except:
        # logging.error("Switched similarity matrix")
        similarity = matrix
    for other_user in problems[problem]['users']:
        try:
            similarity[user][other_user]
            if user != other_user:
                score_sim_sum += (problems[problem]['users']
                                  [other_user] * similarity[user][other_user])
                sim_sum += similarity[user][other_user]
        except:
            error_obj['sim_key_error'] += 1
            # logging.error("Similarity key error (%s), other user: %s, user: %s", str(
            #     error_obj['sim_key_error']), other_user, user)
            continue
    if sim_sum > 0:
        prediction = score_sim_sum / sim_sum
    else:
        prediction = 0
    return prediction


def get_top_problems_per_user(user, matrix=None):
    # generates predicted output for all problems the user did not solve
    predictions[user] = []
    error_obj['sim_key_error'] = 0
    for problem in problems:
        if (not problem in users[user]['problems']) and (len(problem) <= 4):
            computed_p = compute_predictions(user, problem, matrix)
            # push only positive predictions
            if(computed_p > 0):
                predictions[user].append(
                    {'problem': problem, 'prediction': computed_p})
    sorted_pred = sorted(
        predictions[user], key=lambda prob: -prob['prediction'])
    logging.info("Computed predictions for user: %s", user)
    return sorted_pred


def generate_predictions_for_all():
    logging.info("Started generating recommendations for all")
    for user in users:
        if predictions.get(user, None) == None:
            predictions[user] = []
        predictions[user] = get_top_problems_per_user(user)
    logging.info("finished geenrating predictions for all")


def extract_problem_info(problem_id):
    info = {'tags': [], 'level': 'A', 'div': 1}
    if all_problems.get(problem_id):
        problem = all_problems[problem_id]
        info['level'] = list(problem['id'])[-1]
        if problem.get('div', None) != None:
            info['div'] = problem['div']
        for tag in problem['tags']:
            info['tags'].append(tag['tag'])
    else:
        logging.error('Problem not found: %s', problem_id)
        crawl_not_found_problem(problem_id)
    return info


def get_set_stat(problem_set):
    # pred_set = []
    pred_set_stats = {'tags': {}, 'count': 0, 'level': {}}
    for problem in problem_set:
        # pred_set.append(all_problems[problem_id])
        pred_set_stats['count'] += 1
        prob_info = extract_problem_info(problem)
        problem_level = prob_info['level'] + str(prob_info['div'])
        if pred_set_stats['level'].get(problem_level) == None:
            pred_set_stats['level'][problem_level] = 1
        else:
            pred_set_stats['level'][problem_level] += 1
        for tag in prob_info['tags']:
            if pred_set_stats['tags'].get(tag) == None:
                pred_set_stats['tags'][tag] = 1
            else:
                pred_set_stats['tags'][tag] += 1

    # normalizing stats
    if pred_set_stats['count'] > 0:
        for level in pred_set_stats['level']:
            pred_set_stats['level'][level] = pred_set_stats['level'][level] / \
                pred_set_stats['count']
            pred_set_stats['level'][level] = round(
                pred_set_stats['level'][level], 2)
        for tag in pred_set_stats['tags']:
            pred_set_stats['tags'][tag] = pred_set_stats['tags'][tag] / \
                pred_set_stats['count']
            pred_set_stats['tags'][tag] = round(pred_set_stats['tags'][tag], 2)
    # over all difficulty level for this set
    over_all_level = 0
    for level in pred_set_stats['level']:
        if level != None:
            over_all_level = over_all_level + \
                level_to_score(level) * pred_set_stats['level'][level]
    pred_set_stats['overall_level_num'] = over_all_level
    pred_set_stats['overall_level'] = score_to_level(over_all_level)
    return pred_set_stats


def compute_diff(user):
    predicted_ids = [problem['problem'] for problem in predictions[user]]
    stat_1 = get_set_stat(predicted_ids)
    solved_ids = [problem for problem in users[user]['problems']]
    stat_2 = get_set_stat(solved_ids)
    print "pred stats: ", stat_1
    print "solved stat: ", stat_2


def save_recommendations():
    file = open('../JSON/resommendations.json', 'w')
    for user in predictions:
        line_dict = {"user": user, 'problems': predictions[user]}
        line = json.dumps(dict(line_dict)) + "\n"
        file.write(line)
    file.close()


def save_in_DB():
    init_db()
    # collection saves BULK recommendation
    recommendations = db['recommendations']
    new_recommendation = {
        'created': datetime.datetime.utcnow(), 'predections': predictions}
    insert_res = recommendations.insert(new_recommendation, check_keys=False)
    # rec_set_db = recommendations.find_one({"_id": rec_set_id})

    # saving each prediction result per user
    predictions_collection = db['predictions']
    for user in predictions:
        pre_user = predictions_collection.find_one({'user': user})
        line_dict = {"user": user, 'problems': predictions[user]}
        if pre_user == None:
            predictions_collection.insert(line_dict, check_keys=False)
        else:
            predictions_collection.update(
                {'user': user}, line_dict, check_keys=False)
    logging.info("saved all predictions per user")


def filter_path(user, limit=10, path='vertical'):
    analysis = {}
    analysis['predicted_ids'] = predicted_ids = [problem['problem']
                                                 for problem in predictions[user]]
    analysis['predicted_stats'] = get_set_stat(predicted_ids)
    analysis['solved_ids'] = solved_ids = [
        problem for problem in users[user]['problems']]
    analysis['solved_stats'] = get_set_stat(solved_ids)
    if path == 'vertical':
        recommended_set = get_higher_level_set(user, analysis, limit)
    elif path == 'horizontal':
        recommended_set = get_same_level_set(user, analysis, limit)
    elif path == 'next_tag':
        recommended_set = get_tags_next_level(user, analysis, True, limit)
    return recommended_set


def get_higher_level_set(user, analysis, limit=10):

    predicted_ids = analysis['predicted_ids']
    predicted_stats = analysis['predicted_stats']
    solved_ids = analysis['solved_ids']
    solved_stats = analysis['solved_stats']
    next_level = solved_stats['overall_level']
    logging.info("limit: %s", str(limit))
    filtered_set = []
    while len(filtered_set) < limit:
        next_level = score_to_level(level_to_score(next_level) + 1)
        # checking reached max level or not
        for problem_id in predicted_ids:
            logging.info("problem id")
            problem = extract_problem_info(problem_id)
            if(len(filtered_set) < 10):
                filtered_set.append(problem_id)
    return filtered_set


def get_tags_next_level(user, analysis, vertical=True, limit=10):

    predicted_ids = analysis['predicted_ids']
    predicted_stats = analysis['predicted_stats']
    solved_ids = analysis['solved_ids']
    solved_stats = analysis['solved_stats']
    next_level = solved_stats['overall_level']
    # list of sorted tuples according
    tags = solved_stats['tags']
    sorted_tags = sorted(tags.items(), key=operator.itemgetter(1), reverse=vertical)
    # TODO: weighted share
    tag_limit = limit / len(tags)
    filtered_set = []
    trials_count = 0
    logging.info("Started next level per tag recommending for user: %s", user)
    while len(filtered_set) < limit:
        trials_count += 1
        if trials_count > 5:
            logging.error("Reached max trial counts")
            break
        for tag_tuple in sorted_tags:
            tag_text = tag_tuple[0]
            tag_weight = tag_tuple[1]
            tag_share = 0

            # getting most solved level in this tag
            tag_levels = users[user]['tags'][tag_text]
            sorted_tag_levels = sorted(tag_levels.items(),
                                       key=operator.itemgetter(1), reverse=True)
            most_solved_level = sorted_tag_levels[0][0]
            next_tag_level = score_to_level(
                level_to_score(most_solved_level) + 1)
            for problem_id in predicted_ids:
                problem = extract_problem_info(problem_id)
                problem_level = problem['level'] + str(problem['div'])
                if tag_text in problem['tags'] and problem_level == next_tag_level and tag_share < tag_limit:
                    if(len(filtered_set) < 10):
                        tag_share += 1
                        filtered_set.append(problem_id)
    logging.info("Finished next level per tag for user: %s", user)
    return filtered_set


def get_tags_for_level(solved_ids, level):
    tags = []
    for problem_id in solved_ids:
        problem = extract_problem_info(problem_id)
        if problem['level'] == level:
            problem_tags = set(problem['tags'])
            tags_set = set(tags)
            tags_diff = problem_tags - tags_set
            tags = tags + list(tags_diff)
    return tags


def get_same_level_set(user, analysis, limit=10):

    predicted_ids = analysis['predicted_ids']
    predicted_stats = analysis['predicted_stats']
    solved_ids = analysis['solved_ids']
    solved_stats = analysis['solved_stats']
    next_level = solved_stats['overall_level']

    filtered_set = []
    trials_count = 0
    print next_level
    while len(filtered_set) < limit:
        trials_count += 1
        level_tags = get_tags_for_level(solved_ids, next_level)
        if trials_count > 5:
            break
        for problem_id in predicted_ids:
            problem = extract_problem_info(problem_id)
            if (problem['level'] == next_level) and (not set(problem['tags']).issubset(set(level_tags))):
                if(len(filtered_set) < 10):
                    filtered_set.append(problem_id)
        next_level = score_to_level(level_to_score(next_level) + 1)
    return filtered_set


def save_recommendation_in_DB(user, recommendation_set):
    init_db()
    recommendation_collection = db['user_recommendations']
    entry = {'user': user, 'recommendations': recommendation_set}
    recommendation_collection.update(
        {'user': user}, entry, upsert=True, check_keys=False)


def process_new_user(username, domain="localhost:8000", limit=100):
    # handles inserting new user (new crawled user)
    fetch_summarizied_submissions(limit)
    fetch_new_user_from_DB(username)
    load_all_problems()
    init_item_matrix()
    matrix = compute_new_user_sim(username)
    get_top_problems_per_user(username, matrix)
    # TODO: select algorithm based on ranking
    recommended_set = filter_path(username, 10, 'next_tag')
    logging.error("Similarity total errors: %s",
                  str(error_obj['sim_key_error']))
    save_recommendation_in_DB(username, recommended_set)


def compute_new_user_sim(new_user):
    # computes similarity for new user, inserts values in the matrix
    init_db()
    logging.info("Fetching similarity matrix from DB")
    # similarity = db['similarity_matrix'].find_one().sort(
    #     'created', pymongo.ASCENDING).limit(1)
    similarity_item = db['similarity_matrix'].find_one()
    matrix = similarity_item['matrix']
    similarity_id = similarity_item['_id']
    matrix[new_user] = {}
    logging.info("Computing new user similarity")
    for other_user in matrix:
        new_sim = user_user_sim(new_user, other_user, matrix)
        matrix[new_user][other_user] = new_sim
        matrix[other_user][new_user] = new_sim
    logging.info("finished computing new similarity")

    # similarity is the global variable for the sim matrix
    global similarity
    similarity = matrix
    # save new matrix
    logging.info("Saving new similarity matrix")
    new_matrix = {'created': datetime.datetime.utcnow(
    ), 'matrix': matrix, 'users_length': len(users)}
    # db['similarity_matrix'].update({'_id': similarity_id}, new_matrix, upsert=True, check_keys=False)
    return matrix


def fetch_sim_matrix():
    init_db()
    logging.info("Fetching similarity matrix from DB")
    similarity_item = db['similarity_matrix'].find_one()
    return similarity_item


def process_existing_user(username, domain="localhost:8000", limit=10000):
    # handles inserting new user (new crawled user)
    fetch_summarizied_submissions(limit)
    fetch_new_user_from_DB(username)
    load_all_problems(1)
    init_item_matrix()
    matrix = compute_new_user_sim(username)
    get_top_problems_per_user(username, matrix)
    # TODO: select algorithm based on ranking
    recommended_set = filter_path(username, 10, 'next_tag')
    logging.error("Similarity total errors: %s",
                  str(error_obj['sim_key_error']))
    return recommended_set
    # save_recommendation_in_DB(username, recommended_set)


def init_stuff(mongo_option=1, limit=10000):
    logging.info("Initializing...")
    if mongo_option == 1:
        fetch_users_from_DB(limit)
    else:
        process_files()
    load_all_problems(1)
    init_item_matrix()


def crawl_not_found_problem(problem_id):
    scrapyRoot = "https://app.scrapinghub.com/api/run.json"
    scrapyAPIKEY = "32f54a1220b84bbab635e6274271215a"
    scrapyProject = "306033"
    request_body = {
        'spider': 'single_problem',
        'project': scrapyProject,
        'problem_id': problem_id
    }
    headers_body = {
        'Content-Type': 'application/x-www-form-urlencoded'
    }
    params_body = {'apikey': scrapyAPIKEY}
    r = requests.post(scrapyRoot, headers=headers_body,
                      params=params_body, data=request_body)
    msg = "Sent request.\nStatus: ", r.status_code, ", Response: ", r.content
    if r.status_code == 200:
        logging.info(msg)
    else:
        logging.error(msg)
    return


if __name__ == "__main__":
    target_user = "maged918"
    users_limit = 100
    # logging.info("New session started. Ya saaaater! limit: %s", users_limit)
    rec_set = process_existing_user(target_user, None, users_limit)
    print rec_set
    # save_submissions()
    # init_stuff()
    # compute_user_sim()
    # save_problems_in_DB()
    # save_recommendations()
    # compute_diff(target_user)
    # plotting user-user similarity matrix
    # df = pd.DataFrame.from_dict(similarity, orient='index', dtype=None)
    # print df
