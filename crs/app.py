import json
import os
import pandas as pd
from sqlalchemy import exc
from flask_cors import CORS
from flask_script import Manager

from manage import db
from util import load_data
from service import user_service, assesment_service
from encoders.encoder import MyJSONEncoder
from flask_migrate import Migrate, MigrateCommand
from flask import Flask, request, jsonify

# " Global variables for loading graph and data"
# global gb_nodes
# global gb_path
# global gb_job_skill_rel
# global gb_job_skill_rel_tc
# global gb_course_data
# global gb_know_data
# global gb_skill_data
# global gb_courses
# global gb_knowledge
# global gb_skill
# global gb_job_data
# global gb_job_data_tc
# global gb_node_emb
# global gb_ground_truth
# global gb_kc_rel
# global gb_questions
gb_path = "./resources/data/"
gb_job_skill_rel = pd.read_csv(os.path.join(gb_path, 'Job_Skill_Rel.csv'))
gb_course_data = pd.read_csv(os.path.join(gb_path, 'Course_Node.csv'))
gb_courses_desc = pd.read_csv(os.path.join(gb_path, 'Luddy.csv'))
gb_know_data = pd.read_csv(os.path.join(gb_path, 'Knowledge_Node.csv'))
gb_skill_data = pd.read_csv(os.path.join(gb_path, 'Skill_Node.csv'))
gb_nodes = load_data.generate_nodes(gb_course_data, gb_know_data, gb_skill_data, gb_path)
gb_courses = load_data.get_course_data(gb_course_data)
gb_courses_desc = load_data.get_course_desc(gb_courses_desc)
gb_knowledge = load_data.get_know_data(gb_know_data)
gb_skill = load_data.get_skill_data(gb_skill_data)
gb_job_data = load_data.get_job_data(gb_job_skill_rel)
gb_node_emb = load_data.get_node_embeddings(gb_path,'graph__1.emb')
gb_job_skill_rel_tc = pd.read_csv(os.path.join(gb_path,'Job_Skill_Rel_1.csv'))
gb_job_data_tc = load_data.get_job_data(gb_job_skill_rel)
gb_ds_ground_truth = load_data.get_ground_truth(gb_path,'Data_Scientist_GT.xlsx')
gb_kc_rel = load_data.get_kc_relationships(gb_path,'Knowledge_Course_Rel.csv')
gb_questions = pd.read_csv(os.path.join(gb_path,'Question_Node.csv'))
gb_rev_nodes = load_data.generate_rev_nodes(gb_nodes)
gb_ass_matrix = load_data.generate_ass_matrix()

app = Flask(__name__)
app.json_encoder = MyJSONEncoder
app.config['DEBUG'] = True
app.config['SQLALCHEMY_DATABASE_URI'] = "postgresql://postgres:idontknow.3@localhost:5432/crs"
#"postgresql://postgres:idontknow.3@localhost:5438/crs"
#os.environ['SQLALCHEMY_DATABASE_URI']
app.debug = False
CORS(app)
db.init_app(app)
db.app = app
migrate = Migrate(app, db)
manager = Manager(app)
manager.add_command('db', MigrateCommand)

@manager.command
def load_questions():
    load_data.load_questions('Question_Knowledge_Rel.csv', gb_questions, gb_path)


@app.route('/register',methods=["POST"])
def register():

    data = json.loads(request.data)
    user_data = data.get("user",None)
    username = user_data.get("username", None)
    email = user_data.get("email", None)
    saved_user = None
    try :
        saved_user = user_service.register(username,email)
    except exc.SQLAlchemyError:
        print('Could not save the user')

    return jsonify(saved_user)

@app.route('/login',methods=["POST"])
def login():
    data = json.loads(request.data)
    user = data.get("user",None)
    username = user.get("username", None)
    password = user.get("password", None)
    try:
        user = user_service.login(username,password)
    except exc.SQLAlchemyError:
        print('User not found')
    print(jsonify(user))
    return jsonify(user)

@app.route('/start',methods=["POST"])
def start_assesment():
    data = json.loads(request.data)
    ass_data = assesment_service.start_assesment(data)
    return jsonify(ass_data)

@app.route('/nextQuestion',methods=["POST"])
def next_question():
    data = json.loads(request.data)
    ass_data = assesment_service.next_question(data)
    return jsonify(ass_data)

@app.route('/satisfaction',methods=["POST"])
def user_satisfaction():
    data = json.loads(request.data)
    sat_data = assesment_service.user_satisfaction(data)
    return jsonify(sat_data)



@app.route('/recommend',methods=["POST"])
def recommend_courses():
    data = json.loads(request.data)
    sat_data = assesment_service.recommend_courses(data)
    return jsonify(sat_data)

@app.route('/report',methods=["POST"])
def generate_report():
    data = json.loads(request.data)
    sat_data = assesment_service.generate_report(data)
    return jsonify(sat_data)


if __name__ == '__main__':
    manager.run()
