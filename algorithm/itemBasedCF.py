from collections import defaultdict
from operator import itemgetter

from algorithm.similarity import get_item_simi
from utils.common import ModelUtil


class ItemBasedCF:
    def __init__(self, use_iuf_similarity=False):
        print("[INFO] start ItemBasedCF")
        self.use_iuf_similarity = use_iuf_similarity
        self.model_name = 'movie_sim_mat-iif' if self.use_iuf_similarity else 'movie_sim_mat'

    def set_simi_func_name(self, simi_func_name):
        if simi_func_name == "":
            simi_func_name = "log"
        self.simi_func_name = simi_func_name

    def fit(self, trainset):
        print('[INFO] Training a new model...')
        self.trainset = trainset
        self.movie_sim_mat, self.movie_popular = get_item_simi(self.trainset,
                                                               self.use_iuf_similarity,
                                                               self.simi_func_name)

    def loadFromFile(self):
        self.movie_sim_mat = ModelUtil().load(self.model_name)
        self.movie_popular = ModelUtil().load('movie_popular')
        self.trainset = ModelUtil().load('trainset')
        print('[SUCCESS] model is loaded from the file')

    def saveToFile(self):
        ModelUtil().save(self.movie_sim_mat, self.model_name)
        ModelUtil().save(self.movie_popular, 'movie_popular')
        ModelUtil().save(self.trainset, 'trainset')
        print('[SUCCESS] The new model has saved')

    def recommend(self, user):
        predict_score = defaultdict(int)
        for movie, rating in self.trainset[user].items():
            sorted_sim_movie = sorted(self.movie_sim_mat[movie].items(), key=itemgetter(1), reverse=True)
            for related_movie, similarity_factor in sorted_sim_movie[0:20]:
                if related_movie in self.trainset[user]:
                    continue
                predict_score[related_movie] += similarity_factor * rating
        sorted_rec_movie = sorted(predict_score.items(), key=itemgetter(1), reverse=True)
        return [movie for movie, _ in sorted_rec_movie[0:10]]
