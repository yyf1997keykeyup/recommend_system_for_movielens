import random
import time
from collections import defaultdict
from itertools import islice

import yaml

from algorithm.itemBasedCF import ItemBasedCF
from algorithm.userBasedCF import UserBasedCF
from utils.common import ModelUtil, ModelLoadException
from utils.test import TestCase

DATASET_CONFIG = {
    'ml-100k': {"path": 'data/ml-100k/u.data', "sep": '\t'},
    'ml-1m': {"path": 'data/ml-1m/ratings.dat', "sep": '::'},
}

NAME2MODEL = {
    'UserCF': lambda: UserBasedCF(),
    'UserCF-IIF': lambda: UserBasedCF(use_iif_similarity=True),
    'ItemCF': lambda: ItemBasedCF(),
    'ItemCF-IUF': lambda: ItemBasedCF(use_iuf_similarity=True),
}


def run(model_name, dataset, test_ratio, simi_func_name, user_list, random_split_seed):
    print('[INFO] model_name: %s, dataset: %s, test_ratio: %f' % (model_name, dataset, test_ratio))
    model_util = ModelUtil(dataset, test_ratio)
    try:
        train_set = model_util.load('trainset')
        test_set = model_util.load('testset')
    except ModelLoadException:
        dataset_path = DATASET_CONFIG[dataset]["path"]
        dataset_sep = DATASET_CONFIG[dataset]["sep"]
        with open(dataset_path) as f:
            ratings = []
            for line in islice(f, 0, None):
                ratings.append(line.strip('\r\n').split(dataset_sep)[:3])
        print("[SUCCESS] dataset has been loaded")

        train_set, test_set = defaultdict(dict), defaultdict(dict)
        test_actual_size = 0
        random.seed(random_split_seed)
        for user, movie, rate in ratings:
            if random.random() <= test_ratio:
                test_set[user][movie] = int(rate)
                test_actual_size += 1
            else:
                train_set[user][movie] = int(rate)
        print('[SUCCESS] split data has been split to a test set and a train set')
        print('[INFO] the ratio of train set = %s' % (len(ratings) - test_actual_size))
        print('[INFO] the ratio of test set = %s' % test_actual_size)
        model_util.save(train_set, 'trainset')
        model_util.save(test_set, 'testset')

    model = NAME2MODEL[model_name]()
    model.set_simi_func_name(simi_func_name)

    try:
        model.loadFromFile()
    except ModelLoadException:
        print('[INFO] No model saved in model dir')
        model.fit(train_set)

    print('[Success] A new model has been trained')
    model.saveToFile()

    print("[INFO] start to recommend based on the recommend_user_list")
    for user in user_list:
        print("[SUCCESS] For user whose id = %s, The id of the recommended movies:" % user)
        print(model.recommend(str(user)))

    test_case = TestCase(model, test_set)
    test_case.run()
    print('[SUCCESS] precision=%.5f\t recall=%.5f\t coverage=%.5f\t popularity=%.5f\t' %
          (test_case.getPercision(), test_case.getRecall(), test_case.getCoverage(), test_case.getPopularity()))


if __name__ == '__main__':
    start_time = time.time()
    with open("./config.yml") as f:
        config = yaml.safe_load(f)
    run(config['model_type'], config['dataset'], config['test_ratio'],
        config['simi_func_name'], config['recommend_user_list'], config['random_split_seed'])
    print('[INFO] total %.2f seconds have spent' % (time.time() - start_time))
