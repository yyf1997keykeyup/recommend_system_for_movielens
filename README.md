# Recommendation System for MovieLens Dataset

## Python Environment
* python: 3.0 - 3.8

## How to install and run
```shell script
# install PyYAML lib
pip install -r requirements.txt

# (optional) modify config.yml file as you want
vim config.yml

# run the main.py
python main.py
```

## console logging output example:
```shell script
[INFO] model_name: UserCF-IIF, dataset: ml-100k, test_ratio: 0.100000
[SUCCESS] dataset has been loaded
[SUCCESS] split data has been split to a test set and a train set
[INFO] the ratio of train set = 89961
[INFO] the ratio of test set = 10039
[INFO] start UserBasedCF
[INFO] No model saved in model dir
[INFO] Training a new model...
[INFO] building movie-users inverse table
[SUCCESS] building movie-users inverse table success.
[INFO] the number of all movies = 1671
[INFO] calculating users co-rated movies similarity
[INFO] 0 steps, 0.01 seconds have spent
[INFO] 500 steps, 7.86 seconds have spent
[INFO] 1000 steps, 9.55 seconds have spent
[INFO] 1500 steps, 9.74 seconds have spent
[SUCCESS] Got user co-rated movies similarity matrix success, spent 9.739250 seconds
[INFO] calculating user-user similarity matrix
[INFO] 0 steps, 0.00 seconds have spent
[INFO] 500 steps, 0.49 seconds have spent
[SUCCESS] Got user-user similarity matrix success, spent 0.922652 seconds
[Success] A new model has been trained
[SUCCESS] The new model has saved
[INFO] start to recommend based on the recommend_user_list
[SUCCESS] For user whose id = 1, The id of the recommended movies:
['168', '423', '433', '4', '603', '318', '313', '474', '568', '288']
[SUCCESS] For user whose id = 2, The id of the recommended movies:
['124', '303', '750', '515', '248', '9', '237', '181', '744', '137']
[SUCCESS] For user whose id = 3, The id of the recommended movies:
['313', '286', '269', '751', '315', '301', '748', '750', '895', '310']
[SUCCESS] For user whose id = 100, The id of the recommended movies:
['748', '332', '682', '307', '339', '327', '343', '245', '304', '678']
[SUCCESS] For user whose id = 500, The id of the recommended movies:
['64', '132', '318', '427', '12', '732', '7', '14', '173', '22']
[SUCCESS] For user whose id = 900, The id of the recommended movies:
['127', '50', '479', '98', '286', '276', '179', '174', '515', '496']
[INFO] start testing
[INFO] 0 steps, 0.01 seconds have spent..
[INFO] 500 steps, 1.86 seconds have spent..
[SUCCESS] Test recommendation system success, spent 3.154643 seconds
[SUCCESS] precision=0.18876	 recall=0.17731	 coverage=0.24955	 popularity=5.42733	
[INFO] total 14.67 seconds have spent
```
