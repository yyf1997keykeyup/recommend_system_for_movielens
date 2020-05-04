import math
import time
from collections import defaultdict


def simi_log(entities):
    return 1 / math.log(1 + len(entities))


def simi_inverse_proportional(entities):
    return 1 / len(entities)


def simi_quadratic_inverse_proportional(entities):
    return 1 / pow(len(entities), 2)


def simi_sqrt_inverse_proportional(entities):
    return 1 / pow(len(entities), 0.5)


def simi_exponent_inverse_proportional(entities):
    return 1 / math.exp(1 / len(entities))


SIMINAME2FUNC = {
    "log": simi_log,
    "inverse_proportional": simi_inverse_proportional,
    "quadratic_inverse_proportional": simi_quadratic_inverse_proportional,
    "sqrt_inverse_proportional": simi_inverse_proportional,
    "exponent_inverse_proportional": simi_exponent_inverse_proportional,

}


def get_user_simi(trainset, use_iif_similarity=False, simi_func_name="log"):
    # some parts of code are based on the book -- <Recommended system practice> written by Liang Xiang
    movie2users, movie_popular = get_movie2users(trainset)

    print('[INFO] calculating users co-rated movies similarity')
    user_sim_mat = {}
    simi_func = SIMINAME2FUNC[simi_func_name]
    start_time = time.time()
    step_count = 0
    for movie, users in movie2users.items():
        for user1 in users:
            if user1 not in user_sim_mat:
                user_sim_mat[user1] = defaultdict(int)
            for user2 in users:
                if user1 == user2:
                    continue
                elif use_iif_similarity:
                    user_sim_mat[user1][user2] += simi_func(users)
                else:
                    user_sim_mat[user1][user2] += 1

        if step_count % 500 == 0:
            print('[INFO] %d steps, %.2f seconds have spent' % (step_count, time.time() - start_time))
        step_count += 1
    print('[SUCCESS] Got user co-rated movies similarity matrix success, spent %f seconds' % (time.time() - start_time))

    print('[INFO] calculating user-user similarity matrix')
    start_time = time.time()
    step_count = 0
    for user1, related_users in user_sim_mat.items():
        for user2, count in related_users.items():
            user_sim_mat[user1][user2] = count / math.sqrt(len(trainset[user1]) * len(trainset[user2]))
        if step_count % 500 == 0:
            print('[INFO] %d steps, %.2f seconds have spent' % (step_count, time.time() - start_time))
        step_count += 1

    print('[SUCCESS] Got user-user similarity matrix success, spent %f seconds' % (time.time() - start_time))
    return user_sim_mat, movie_popular


def get_item_simi(trainset, use_iuf_similarity=False, simi_func_name="log"):
    # some parts of code are based on the book -- <Recommended system practice> written by Liang Xiang
    _, movie_popular = get_movie2users(trainset)

    print('[INFO] calculating items co-rated similarity matrix')
    movie_sim_mat = {}
    simi_func = SIMINAME2FUNC[simi_func_name]
    start_time = time.time()
    step_count = 0

    for user, movies in trainset.items():
        for movie1 in movies:
            if movie1 not in movie_sim_mat:
                movie_sim_mat[movie1] = defaultdict(int)
            for movie2 in movies:
                if movie1 == movie2:
                    continue
                elif use_iuf_similarity:
                    movie_sim_mat[movie1][movie2] += simi_func(movies)
                else:
                    movie_sim_mat[movie1][movie2] += 1
        if step_count % 500 == 0:
            print('[INFO] %d steps, %.2f seconds have spent' % (step_count, time.time() - start_time))
        step_count += 1
    print('[SUCCESS] Got items co-rated similarity matrix success, spent %f seconds' % (time.time() - start_time))

    print('[INFO] calculating item-item similarity matrix')
    start_time = time.time()
    step_count = 0
    for movie1, related_items in movie_sim_mat.items():
        for movie2, count in related_items.items():
            movie_sim_mat[movie1][movie2] = count / math.sqrt(movie_popular[movie1] * movie_popular[movie2])
        if step_count % 500 == 0:
            print('[INFO] %d steps, %.2f seconds have spent' % (step_count, time.time() - start_time))
        step_count += 1

    print('[SUCCESS] Got item-item similarity matrix success, spent %f seconds' % (time.time() - start_time))
    return movie_sim_mat, movie_popular


def get_movie2users(trainset):
    print('[INFO] building movie-users inverse table')
    movie2users = defaultdict(set)
    movie_popular = defaultdict(int)

    for user, movies in trainset.items():
        for movie in movies:
            movie2users[movie].add(user)
            movie_popular[movie] += 1
    print('[SUCCESS] building movie-users inverse table success.')
    print('[INFO] the number of all movies = %d' % len(movie2users))
    return movie2users, movie_popular
