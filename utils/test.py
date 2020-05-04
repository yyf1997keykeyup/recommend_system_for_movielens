import math
import time


class TestCase:
    def __init__(self, model, testset):
        self.model = model
        self.testset = testset

        self.hit = 0
        self.rec_count = 0
        self.test_count = 0
        self.rec_movies = set()
        self.movie_popular_sum = 0

    def run(self):
        print('[INFO] start testing')
        start_time = time.time()
        step_count = 0

        for user in self.model.trainset:
            test_movies = self.testset.get(user, {})
            rec_movies = self.model.recommend(user)
            for rec_movie in rec_movies:
                if rec_movie in test_movies:
                    self.hit += 1
                self.rec_movies.add(rec_movie)
                self.movie_popular_sum += math.log(1 + self.model.movie_popular[rec_movie])

            self.rec_count += 10  # todo: the same as the length of sorted_rec_movie in the recommend method
            self.test_count += len(test_movies)

            if step_count % 500 == 0:
                print('[INFO] %d steps, %.2f seconds have spent..' % (step_count, time.time() - start_time))
            step_count += 1

        print('[SUCCESS] Test recommendation system success, spent %f seconds' % (time.time() - start_time))

    def getPercision(self):
        return self.hit / (1.0 * self.rec_count)

    def getRecall(self):
        return self.hit / (1.0 * self.test_count)

    def getCoverage(self):
        return len(self.rec_movies) / (1.0 * len(self.model.movie_popular))

    def getPopularity(self):
        return self.movie_popular_sum / (1.0 * self.rec_count)
