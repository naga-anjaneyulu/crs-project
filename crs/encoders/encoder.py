from flask import jsonify
from flask.json import JSONEncoder

from model.models import User, Job, AssesmentData, Question, UserSatisfaction


class MyJSONEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, User):
            return {
                'id': obj.id,
                'username': obj.username,
                'email': obj.email,

            }

        if isinstance(obj, Job):
            return {
                'id': obj.id,
                'user_id': obj.user_id,
                'category1': obj.category1,
                'category2': obj.category2,
                'category3': obj.category3,

            }
        if isinstance(obj, Question):
            return {
                 'ques_id' : obj.ques_id,
                 'question': obj.question,
                 'answer' : obj.answer,
                 'level' : obj.level,
                 'know' : obj.know
            }

        if isinstance(obj, AssesmentData):
            return {

                'user_id': obj.user_id,
                'job_id': obj.job_id,
                'count' : obj.count,
                'response' : obj.response,
                'pseudo' : obj.pseudo,
                'category1': obj.category1,
                'category2': obj.category2,
                'category3': obj.category3
            }

        if isinstance(obj, UserSatisfaction):
            return {

                'user_id': obj.user_id,
                'pseudo_recommend': obj.pesudo_recommend,
                'category' : obj.category,
                'refine_quality': obj.refine_quality,
                'refine_recommend': obj.refine_recommend,
                'not_recommend' : obj.not_recommend,
                'category1': obj.category1,
                'category2': obj.category2,
                'category3': obj.category3,
                'gt':obj.gt,
                'count' : obj.count

            }
        return super(MyJSONEncoder, self).default(obj)