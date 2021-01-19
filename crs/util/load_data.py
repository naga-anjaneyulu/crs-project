import os
import csv
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from manage import db
from model.models import Question

"@Get Course data."
def get_course_data(course_data):
    courses = {}
    for index, row in course_data.iterrows():
        courses[row['CourseID']] = row['Course Name']
    return courses


"@Get Knowledge data."
def get_know_data(know_data):
    know = {}
    for index, row in know_data.iterrows():
        know[row['Knowledge Name']] = row['KnowledgeID']
    return know


"@Get Skill data."
def get_skill_data(skill_data):
    skill = {}
    for index, row in skill_data.iterrows():
        skill[row['Skill Name']] = row['SkillID']
    return skill


"@Mapping job categories to the skills required. "
def get_job_data(job_skill_rel):
    job_data = {}
    for index, row in job_skill_rel.iterrows():
        row['Skill'] = row['Skill'].replace("\n", "").replace(",", "").replace("\r", "").replace("(", "").replace(")",
                                                                                                                  "").replace(
            "-", "").lower().strip()
        row['Category'] = row['Category'].replace("\n", "").replace(",", "").replace("\r", "").replace("(", "").replace(
            ")", "").replace("-", "").lower().strip()

        if row['Category'] in job_data.keys() and row['Skill'] not in job_data[row['Category']]:
            job_data[row['Category']] += [row['Skill']]
        else:
            job_data[row['Category']] = [row['Skill']]
    return job_data


"@Generating all the nodes in the graph. "
def generate_nodes(course_data, know_data, skill_data, path):
    nodes = {}
    count = 1

    for index, row in course_data.iterrows():
        nodes[str(row['CourseID']).strip()] = count
        count += 1

    for index, row in know_data.iterrows():
        nodes[str(row['KnowledgeID']).strip()] = count
        count += 1

    for index, row in skill_data.iterrows():
        nodes[str(row['SkillID']).strip()] = count
        count += 1

    return nodes


"@Generating nodes --> nodeid. "
def generate_rev_nodes(nodes):
    rev_nodes = {}

    for key, value in nodes.items():
        rev_nodes[value] = key

    return rev_nodes


"@Loading all the node embeddings."
def get_node_embeddings(path, filename):
    embeddings = {}
    f = open(os.path.join(path, filename), "r")
    for line in f:
        embs = line.split()
        new_embs = [float(val) for val in embs]
        if len(embs) > 2:
            embeddings[int(new_embs[0])] = new_embs[1:]
    f.close()

    return pd.DataFrame.from_dict(embeddings)


" @Loading all the ground truth"
def get_ground_truth(path,filename):
    truth = pd.read_excel(os.path.join(path, filename))
    return truth


" @Loading knowledge course relationships"
def get_kc_relationships(path,filename):
    know_course_rel = pd.read_csv(os.path.join(path,filename))
    kc_rel ={}
    for index,rows in know_course_rel.iterrows():
        if rows["To CourseID"] in kc_rel.keys():
            kc_rel[rows["To CourseID"]] += [rows["From KnowledgeID"]]
        else:
             kc_rel[rows["To CourseID"]] = [rows["From KnowledgeID"]]
    return kc_rel

"  @Loading questions into the database"
def load_questions(filename,gb_questions,gb_path):
    qk_rel_data = pd.read_csv(os.path.join(gb_path,filename))
    qk_rel ={}
    for index,rows in qk_rel_data.iterrows():
        if rows["From QuestionID"] in qk_rel.keys() :
            qk_rel[rows["From QuestionID"]] += [rows["To KnowledgeID"]]
        else:
            qk_rel[rows["From QuestionID"]] = [rows["To KnowledgeID"]]

    for index, rows in gb_questions.iterrows():
        if rows["QuestionID"] in qk_rel :
            know_list = list(set(qk_rel[rows["QuestionID"]]))
            for know in know_list:
                ques = Question(rows["QuestionID"],rows["Question"],rows["Answer"],rows["Difficulty"],know)
                db.session.add(ques)
                db.session.commit()


"@Generating nodes --> nodeid. "
def generate_rev_nodes(nodes):
    rev_nodes = {}
    for key, value in nodes.items():
        rev_nodes[value] = key

    return rev_nodes

"@Generating course description"
def get_course_desc(gb_course_desc):
    courses = {}
    for index, row in gb_course_desc.iterrows():
        courses[row['CourseID'].strip()] = row['Description']
    return courses


def generate_ass_matrix():
    ass_matrix = {}
    ass_matrix["Software Developer"] = "software_developer_ass.csv"
    ass_matrix["Data Scientist"] = "data_scientist_ass.csv"
    ass_matrix["Cyber Secuirty Analyst"] = "cyber_security_analyst_ass.csv"
    ass_matrix["Database Developer"] = "database_developer_ass.csv"
    return ass_matrix