from collections import defaultdict
from operator import itemgetter

from algorithm.similarity import get_user_simi
from utils.common import ModelUtil


class UserBasedCF:
    def __init__(self, use_iif_similarity=False):
        print("[INFO] start UserBasedCF")
        self.use_iif_similarity = use_iif_similarity
        self.model_name = 'user_sim_mat-iif' if self.use_iif_similarity else 'user_sim_mat'

    def set_simi_func_name(self, simi_func_name):
        if simi_func_name == "":
            simi_func_name = "log"
        self.simi_func_name = simi_func_name

    def fit(self, trainset):
        print('[INFO] Training a new model...')
        self.trainset = trainset
        self.user_sim_mat, self.movie_popular = get_user_simi(self.trainset,
                                                              self.use_iif_similarity,
                                                              self.simi_func_name)

    def loadFromFile(self):
        self.user_sim_mat = ModelUtil().load(self.model_name)
        self.movie_popular = ModelUtil().load('movie_popular')
        self.trainset = ModelUtil().load('trainset')
        print('[SUCCESS] model is loaded from the file')

    def saveToFile(self):

        ModelUtil().save(self.user_sim_mat, self.model_name)
        ModelUtil().save(self.movie_popular, 'movie_popular')
        ModelUtil().save(self.trainset, 'trainset')
        print('[SUCCESS] The new model has saved')

    def recommend(self, user):
        predict_score = defaultdict(int)
        sorted_sim_user = sorted(self.user_sim_mat[user].items(), key=itemgetter(1), reverse=True)
        for similar_user, similarity_factor in sorted_sim_user[0:20]:
            for movie, rating in self.trainset[similar_user].items():
                if movie in self.trainset[user]:
                    continue
                predict_score[movie] += similarity_factor * rating
        sorted_rec_movie = sorted(predict_score.items(), key=itemgetter(1), reverse=True)
        return [movie for movie, _ in sorted_rec_movie[0:10]]
