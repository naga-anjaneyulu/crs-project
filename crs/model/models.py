import pandas as pd
from sqlalchemy import Column, Integer, ForeignKey
from sqlalchemy.orm import relationship, backref

from manage import db


class User(db.Model):
    __tablename__ = 'user'
    id = db.Column(db.Integer(), primary_key=True)
    username = db.Column(db.String(255))
    email = db.Column(db.String(255))


    def __init__(self,username,email):
        self.username = username
        self.email = email
    def get_username(self):
        return self.username
    def get_email(self):
        return self.email
    def get_id(self):
        return self.id


class Job(db.Model):
    __tablename__ = 'job'
    id = Column(db.Integer(), primary_key=True)
    user_id = Column(db.Integer(), ForeignKey('user.id'))
    category1 = db.Column(db.String(80))
    category2 = db.Column(db.String(80))
    category3 = db.Column(db.String(80))
    user = relationship("User", backref=backref("user", uselist=False))


    def __init__(self,user_id,category1,category2,category3):
        self.user_id = user_id
        self.category1 = category1
        self.category2 = category2
        self.category3 = category3

    def get_category1(self):
        return self.category1
    def get_category2(self):
        return self.category2
    def get_category3(self):
        return self.category3
    def get_user_id(self):
        return self.user_id



class Response(db.Model):
    __tablename__ = 'response'
    id = Column(db.Integer(), primary_key=True)
    user_id = Column(db.Integer(), ForeignKey('user.id'))
    course_id = db.Column(db.String(255))
    course_name = db.Column(db.String(255))
    response =  db.Column(db.String(255))
    recommend =db.Column(db.Integer())
    category = db.Column(db.String(255))
    user1 = relationship("User", backref=backref("user1", uselist=False))

    def __init__(self,user_id,course_id,course_name,response,recommend,category):
        self.user_id = user_id
        self.course_id = course_id
        self.response = response
        self.recommend = recommend
        self.course_name = course_name
        self.category = category


class UserSatisfaction(db.Model):
    __tablename__ = 'user_satisfaction'
    id = Column(db.Integer(), primary_key=True)
    user_id = Column(db.Integer(), ForeignKey('user.id'))
    refine_quality = db.Column(db.String(255))
    category = db.Column(db.String(255))
    user3 = relationship("User", backref=backref("user3", uselist=False))
    def __init__(self, user_id):
        self.user_id = user_id
        self.pesudo_recommend = {}
        self.refine_recommend ={}
        self.refine_quality =""
        self.not_recommend ={}
        self.gt ={}
        self.count = 0
        self.category = ""
        self.category1 = ""
        self.category2 = ""
        self.category3 = ""


    def get_category(self):
        return self.category
    def set_category1(self,category):
        self.category = category
    def set_count(self,count):
        self.count = count
    def get_count(self):
        return self.count
    def get_user_id(self):
        return self.user_id
    def set_user_id(self,user_id):
        self.user_id = user_id
    def get_refine_quality(self):
        return self.refine_quality
    def set_refine_quality(self,refine_quality):
        self.refine_quality = refine_quality
    def get_pseudo_recommend(self):
        return self.pesudo_recommend
    def set_pseudo_recommend(self,pseudo_recommend):
        self.pesudo_recommend = pseudo_recommend
    def get_refine_recommend(self):
        return self.refine_recommend
    def set_refine_recommend(self,refine_recommend):
        self.refine_recommend = refine_recommend
    def get_not_recommend(self):
        return self.not_recommend
    def set_not_recommend(self, not_recommend):
        self.not_recommend = not_recommend
    def get_gt(self):
        return self.gt
    def set_gt(self, gt):
        self.gt = gt
    def get_category1(self):
        return self.category1
    def get_category2(self):
        return self.category2
    def get_category3(self):
        return self.category3
    def set_category1(self,category1):
        self.category1 = category1
    def set_category(self,category):
        self.category = category
    def set_category2(self, category2):
        self.category2 = category2
    def set_category3(self, category3):
        self.category3 = category3

class AssesmentData():

    def __init__(self,userid,jobid,category1,category2,category3):
        self.user_id = userid
        self.job_id = jobid
        self.count =0
        self.response=""
        self.category1 = category1
        self.category2 = category2
        self.category3 = category3
        self.pseudo ={}
    def get_pesudo(self):
        return self.pseudo
    def set_pseudo(self,pseudo):
        self.pseudo = pseudo;
    def get_category1(self):
        return self.category1
    def get_category3(self):
        return self.category3
    def get_category2(self):
        return self.category2
    def set_count(self,count):
        self.count = count
    def get_count(self):
        return self.count
    def set_response(self,response):
        self.response=response
    def get_response(self):
        return self.response
    def get_user_id(self):
        return self.user_id


class GroundTruth(db.Model):
    _tablename__ = 'ground_truth'
    id = Column(db.Integer(), primary_key=True)
    user_id = Column(db.Integer(), ForeignKey('user.id'))
    course_id =  course_id = db.Column(db.String(255))
    course_name = db.Column(db.String(255))
    choice =  db.Column(db.String(255))
    gt = db.Column(db.Integer())
    def __init__(self, user_id, course_id, course_name, choice, gt):
        self.user_id = user_id
        self.course_id = course_id
        self.course_name = course_name
        self.choice = choice
        self.gt = gt
    def get_user_id(self):
        return self.user_id
    def get_course_id(self):
        return self.course_id
    def get_course_name(self):
        return self.course_name
    def get_choice(self):
        return self.choice
    def get_gt(self):
        return self.gt
    def set_user_id(self,user_id):
        self.user_id = user_id
    def set_course_id(self,course_id):
        self.course_id = course_id
    def set_course_name(self,course_name):
        self.course_name = course_name
    def set_choice(self,choice):
        self.choice = choice
    def set_gt(self,gt):
        self.gt = gt

class Question(db.Model):
    __tablename__ = 'question'
    id = db.Column(db.Integer(), primary_key=True)
    ques_id = db.Column(db.String(255))
    question = db.Column(db.String(255000))
    answer = db.Column(db.String(255000))
    know =  db.Column(db.String(255000))
    level = db.Column(db.String(255000))

    def __init__(self,ques_id,question,answer,level,know):
        self.ques_id = ques_id
        self.question = question
        self.answer = answer
        self.level = level
        self.know = know

    def get_ques_id(self):
        return self.ques_id
    def get_question(self):
        return self.question
    def get_answer(self):
        return self.answer
    def get_id(self):
        return self.id
    def get_know(self):
        return self.know
    def get_level(self):
        return self.level