from pymongo import MongoClient


class Service:

    def __init__(self):
        self.client = MongoClient(
            'mongodb+srv://admin:admin@hippo-cluster-gya0k.mongodb.net/hippo-survey-db?retryWrites=true')
        self.db = self.client['hippo-survey-db']

    def get_all_surveys(self):
        users = self.db['users']
        cursor = users.find({})
        result = [x for x in cursor]

        return str(result)
