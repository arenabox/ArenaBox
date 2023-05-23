import json
import os
import time
import urllib
from collections import defaultdict
from os import listdir
from os.path import isfile, join
from pathlib import Path
from urllib.error import HTTPError

import argparse
import snscrape.modules.twitter as sntwitter
import yaml
from tqdm import tqdm


def from_users(users, save_folder, config_data):
    """
    This method will extract tweets for given list of users.
    :param users: list of users to mine for
    :param save_folder: path to the folder where mined tweets will be saved in json format : save_folder/USER.json
    :param config_data: configuration file dictionary
    :return: Saves the extracted tweets in the save_folder in the format
            tweet_id : {  tweet_info_dict }, ....
    """
    for user in users:
        Path(save_folder).mkdir(parents=True, exist_ok=True)
        name = join(save_folder, f'{user}.json')
        if os.path.exists(name):
            print(f'{user} tweets already crawled.')
            continue
        print(f'Starting tweet crawling for {user}')

        with open(name, "w+") as fd:
            # subprocess.run(["snscrape", "-sleep","0.2","--jsonl", "--progress", "twitter-user", community], stdout=fd)
            data = dict()
            query = f'from:{user}'
            if "since" in config_data:
                query += f' since:{config_data["since"]}'
            if "until" in config_data:
                query += f' until:{config_data["until"]}'
            for i, tweet in enumerate(
                    sntwitter.TwitterSearchScraper(query).get_items()):
                data[tweet.id] = json.loads(tweet.json())
            json.dump(data, fd, indent=4)
        fd.close()
        print(f'Tweet crawl for {user} completed')


def mentioning_users(users, save_folder, config_data):
    """
    This method will extract tweets which has mentioned the given list of users.
    :param users: list of users to mine for
    :param save_folder: path to the folder where mined tweets will be saved in json format : save_folder/@USER.json
    :param config_data: configuration file dictionary
    :return: Saves the extracted tweets in the save_folder in the format
            tweet_id : {  tweet_info_dict }, ....
    """
    for user in users:
        Path(save_folder).mkdir(parents=True, exist_ok=True)
        name = join(save_folder, f'@{user}.json')
        if os.path.exists(name):
            print(f'Mentioned tweets for {user} already crawled.')
            continue
        print(f'Starting crawling for tweets mentioning {user}')

        with open(name, "w+") as fd:
            # subprocess.run(["snscrape", "-sleep","0.2","--jsonl", "--progress", "twitter-user", community], stdout=fd)
            data = dict()
            query = f'@{user}'
            if "since" in config_data:
                query += f' since:{config_data["since"]}'
            if "until" in config_data:
                query += f' until:{config_data["until"]}'
            for i, tweet in enumerate(
                    sntwitter.TwitterSearchScraper(query).get_items()):
                data[tweet.id] = json.loads(tweet.json())
            json.dump(data, fd, indent=4)
        fd.close()
        print(f'Tweet crawl for mentioned {user} completed')


def containing_keyword(save_folder, config_data):
    """
    This method will extract tweets which contains the list of keywords given in the config.
    :param save_folder: path to the folder where mined tweets will be saved in json format : save_folder/KEYWORD.json
    :param config_data: configuration file dictionary

    """
    keywords = config_data['containing_keywords']
    for keyword in keywords:
        Path(save_folder).mkdir(parents=True, exist_ok=True)
        name = join(save_folder, f'{keyword}.json')
        if os.path.exists(name):
            print(f'Mentioned tweets for {keyword} already crawled.')
            continue
        print(f'Starting crawling for tweets mentioning {keyword}')

        with open(name, "w+") as fd:
            # subprocess.run(["snscrape", "-sleep","0.2","--jsonl", "--progress", "twitter-user", community], stdout=fd)
            data = dict()
            query = f'{keyword}'
            if "since" in config_data:
                query += f' since:{config_data["since"]}'
            if "until" in config_data:
                query += f' until:{config_data["until"]}'
            for i, tweet in enumerate(
                    sntwitter.TwitterSearchScraper(query).get_items()):
                data[tweet.id] = json.loads(tweet.json())
            json.dump(data, fd, indent=4)
        fd.close()
        print(f'Tweet crawl for mentioned {keyword} completed')

def _get_mentions_list_for_users(users, save_folder):
    """
    This method extract list of users mentioned by given users. It creates a mapping from a user to its mentioned user
    along with the number of times, user mentioned another user.
    :param users: list of users for which mention users is to be extracted
    :param save_folder: path to folder where tweet data is saved for list of user.
    :return: Mapping from user to mentioned users based on the frequency of mentions.
    """
    nq_list = users
    mentioned_users = defaultdict(lambda: defaultdict(int))
    for user in users:
        name = join(save_folder, f'{user}.json')
        with open(name, "r") as fd:
            user_tweet_data = json.load(fd)
        fd.close()
        for id, tweet in user_tweet_data.items():
            if tweet["mentionedUsers"] is None:
                continue
            for mentioned_user in tweet["mentionedUsers"]:
                mentioned_users[user][mentioned_user['username']] += 1
    for k, v in mentioned_users.items():
        for mentioned_user in list(v.keys()):
            if mentioned_user in nq_list:
                del mentioned_users[k][mentioned_user]
                continue
            nq_list.append(mentioned_user)
    for k, v in mentioned_users.items():
        mentioned_users[k] = dict(sorted(v.items(), key=lambda x: x[1], reverse=True))
        # mentioned_users[community] = dict(sorted(mentioned_users[community].items(), key= lambda x: x[1], reverse=True ))

    return mentioned_users


def from_mentioned_user(users, save_folder, config_data):
    """
    This method extracts tweets for mentioned users for a given list of users. It only extracts tweets for top
    (top_n_users) frequently mentioned users, and then it saves tweets for only those mentioned users who have mentioned
    original users upto a threshold value (interaction_level).
    :param user: username whose mentioned users needs to be crawled
    :param save_folder: path to save mentioned_users tweet crawl as {save_folder}/mentioned_by_{user}.json
    :param config_data: configuration file dictionary

    """
    mentioned_users = _get_mentions_list_for_users(users,save_folder)
    top_n_users = 400
    interaction_level = 3
    if config_data['mentioned_by_params']['top_n_users'] is not None:
        top_n_users = config_data['mentioned_by_params']['top_n_users']
    if config_data['mentioned_by_params']['interaction_level'] is not None:
        interaction_level = config_data['mentioned_by_params']['interaction_level']

    for user, ment_users in mentioned_users.items():
        data = defaultdict(lambda: defaultdict(list))
        print(f"Tweet Crawl started for {user}'s mentioned users")
        for e, (mentioned_user, count) in tqdm(enumerate(ment_users.items())):
            if len(data) == top_n_users:
                break
            query = f'from:{mentioned_user}'
            if "since" in config_data:
                query += f' since:{config_data["since"]}'
            if "until" in config_data:
                query += f' until:{config_data["until"]}'
            for i, tweet in enumerate(
                    sntwitter.TwitterSearchScraper(query).get_items()):
                data[mentioned_user][tweet.id] = json.loads(tweet.json())

            # Code to remove user with low interaction
            c = 0
            valid_user = False
            for _, tweet in data[mentioned_user].items():
                if c >= interaction_level:
                    valid_user = True
                    break
                if tweet["mentionedUsers"] is None:
                    continue
                for user_ in tweet["mentionedUsers"]:
                    if user_['username'] == user:
                        c += 1

            if not valid_user:
                del data[mentioned_user]

        with open(f'{save_folder}/mentioned_by_{user}.json', "w+") as fd:
            json.dump(data, fd, indent=4)
        fd.close()
        print(f'Tweet crawl for {user}\'s mentioned users completed')


def get_tweet_images(save_folder):
    """
    This method can be used to extract images from the tweets. It will extract images for all the tweet json files
    existing in the given folder.
    :param save_folder: path where tweet jsons are stored
    :return: save images to the folder : {save_folder}/images/{user}/IMAGE.jpg
    """
    tweet_json_files = [f for f in listdir(save_folder) if
                      isfile(join(save_folder, f)) and f.endswith('json')]
    for json_file in tweet_json_files:
        user = json_file.split('.')[0]
        print(f'Extracting images for {user}')
        with open(join(save_folder,json_file), "r") as fd:
            user_data_json = json.load(fd)
        fd.close()
        skipped = 0

        for user_name, tweets in tqdm(user_data_json.items()):
            user_images_path = join(save_folder, 'images', user, user_name)
            Path(user_images_path).mkdir(parents=True, exist_ok=True)
            user_image_files = set([f.split('_')[0] for f in listdir(user_images_path) if f.endswith('jpg')])
            for id, tweet in tqdm(tweets.items()):
                media = tweet['media']
                if id in user_image_files or media is None:
                    continue
                for i, m in enumerate(media):
                    try:
                        if m['_type'] == 'snscrape.modules.twitter.Photo':
                            image_name = f'{id}_{i}.jpg'
                            image_path = join(user_images_path, image_name)
                            urllib.request.urlretrieve(m['previewUrl'], image_path)
                            time.sleep(1)
                        elif m['_type'] == 'snscrape.modules.twitter.Video':
                            image_name = f'{id}_{i}.jpg'
                            image_path = join(user_images_path, image_name)
                            urllib.request.urlretrieve(m['thumbnailUrl'], image_path)
                            time.sleep(1)
                    except HTTPError:
                        skipped+=1
                        time.sleep(30)
                        continue
        #print(f'Skipped {skipped} images due to HTTP Error.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for tweets mining')

    parser.add_argument(
        '-config',
        help='Path to data collection config file for tweets',
        type=str, default='./configs/data/collection/tweet.yaml',
    )

    args, remaining_args = parser.parse_known_args()

    with open(args.config) as file:
        config_data = yaml.safe_load(file)

    with open('./configs/base_path.yaml') as file:
        base_path = yaml.safe_load(file)

    save_folder = join(base_path['data']['collection'], config_data['type'])
    users = config_data['users']

    if config_data['from_users']:
        from_users(users,save_folder, config_data)
    if config_data['have_user_mentioned']:
        mentioning_users(users,save_folder, config_data)
    if 'containing_keywords' in config_data:
        containing_keyword(save_folder, config_data)
    if config_data['from_mentioned_users']:
        from_mentioned_user(users, save_folder, config_data)
    if config_data['get_images']:
        get_tweet_images(save_folder)

