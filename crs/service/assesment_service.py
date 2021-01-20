#############################
#  Author : Naga Anjaneyulu #
#         IU Research       #
############################
import math
import os
import csv
import copy
import random
import numpy as np
import pandas as pd
from manage import db
import matplotlib.pyplot as plt
from app import gb_job_skill_rel, gb_node_emb, gb_nodes, gb_job_skill_rel_tc, gb_rev_nodes, gb_ds_ground_truth, \
    gb_courses_desc, gb_path, gb_ass_matrix
from app import  gb_courses, gb_skill, gb_knowledge, gb_kc_rel
from model.models import Job, User, AssesmentData, Question, Response, UserSatisfaction, GroundTruth
from sklearn.metrics.pairwise import cosine_similarity


"@ function for building assesment object"
def build_assesment_data(data):
    job_data = data.get("job", None)
    user = data.get("user", None)
    saved_user = User.query.filter_by(id=user.get("id", None)).first()
    job = Job(saved_user.get_id(), job_data.get("category1"), job_data.get("category2"), job_data.get("category3"))
    job.user = saved_user
    db.session.add(job)
    db.session.commit()
    saved_job = Job.query.filter_by(user_id=job.get_user_id()).first()
    ass_data = AssesmentData(saved_user.get_id(), saved_job.id, saved_job.category1, saved_job.category2, saved_job.category3)
    return ass_data

"@ function for parse assesment object"
def parse_assesment_data(data):
    ass_data = AssesmentData(data.get("user_id",None), data.get("job_id",None),
                             data.get("category1",None), data.get("category2",None) ,data.get("category3",None))
    ass_data.set_count(data.get("count",0))
    ass_data.set_response(data.get("response","NA"))
    ass_data.set_pseudo(data.get("pseudo",{}))
    return ass_data

""" 
    @function for Generating Assessment Matrix. n x m . 
    n : number of courses , m : number of skills listed 
    under the job category / required skills for the job .
"""
def get_assesment_matrix(node_emb, nodes, courses, skills, know, skill_list):
    ""
    n = len(courses)
    m = len(skill_list)
    rows = []
    cols = []
    for course in courses.keys():
        rows += [nodes[course]]
    for skill in skill_list:
        cols += [nodes[skills[skill]]]
    data = np.full((n, m), 0, dtype=int)
    ass_matrix = pd.DataFrame(data,
                              index=pd.Index(rows, name="Courses"),
                              columns=pd.Index(cols, name="Skills"))
    "Assesment Matrix Calculation"
    for col in cols:
        denom = 0
        for row in rows:
            if int(row) in node_emb.columns and int(col) in node_emb.columns:
                denom += \
                (cosine_similarity(node_emb[int(row)].values.reshape(1, -1), node_emb[int(col)].values.reshape(1, -1)))[
                    0][0]
        for row in rows:
            numer = 0
            if int(row) in node_emb.columns and int(col) in node_emb.columns:
                numer = \
                (cosine_similarity(node_emb[int(row)].values.reshape(1, -1), node_emb[int(col)].values.reshape(1, -1)))[
                    0][0]
                if denom != 0:
                    ass_matrix.loc[row, col] = numer / denom
    print("Now sorting")
    ass_matrix = ass_matrix.loc[(ass_matrix.sum(axis=1) != 0), (ass_matrix.sum(axis=0) != 0)]
    for col in ass_matrix.columns:
        ass_matrix = ass_matrix.sort_values(by=col, axis=0, ascending=False, inplace=False, kind='quicksort')
        count = 0
        for index, row in ass_matrix.iterrows():
            if count < 20:
                ass_matrix.at[index, col] = 1
                count += 1
            else:
                ass_matrix.at[index, col] = 0
    ass_matrix = ass_matrix.loc[(ass_matrix.sum(axis=1) != 0), (ass_matrix.sum(axis=0) != 0)]
    print("Initialized assesment matrix")
    return ass_matrix





" @function for initializing reliability matrix"
def initialize_reliability(ass_matrix):
    data = np.full((ass_matrix.shape[0], 2), 0, dtype=np.float64)
    reliability = pd.DataFrame(data,
                               index=pd.Index(ass_matrix.index, name="Courses"),
                               columns=pd.Index(["pi0", "pi1"], name="Probability"))
    for index, row in ass_matrix.iterrows():
        c0 = 0 ;c1 = 0
        for val in row:
            if val == 0:c0 += 1
            elif val == 1:c1 += 1
        reliability.at[index, "pi0"] = c0 / len(row)
        reliability.at[index, "pi1"] = c1 / len(row)

    return reliability

" @function for initializing validity matrix"
def initialize_validity(ass_matrix):
    data = np.full((ass_matrix.shape[1], 4), 0, dtype=np.float64)
    validity = pd.DataFrame(data,
                            index=pd.Index(ass_matrix.columns, name="Skill"),
                            columns=pd.Index(["q00", "q01", "q10", "q11"], name="Probability"))

    # Calculating pseudo recommendation for all the courses.
    ass_matrix['sum'] = ass_matrix[list(ass_matrix.columns)].sum(axis=1)
    ass_matrix = ass_matrix.sort_values(by='sum', axis=0, ascending=False, inplace=False, kind='quicksort')
    count = 0
    pseudo = {}
    for index, rows in ass_matrix.iterrows():
        if (count < 21):
            pseudo[index] = 1
        else:
            pseudo[index] = 0
        count += 1

    for index, rows in validity.iterrows():
        denom_0 = 0;num_00 = 0;num_01 = 0;
        denom_1 = 0;num_10 = 0;num_11 = 0;
        for key, value in pseudo.items():
            num_00 += ((1 if value == 0 else 0) and (1 if ass_matrix.at[key, index] == 0 else 0))
            num_01 += ((1 if value == 0 else 0) and (1 if ass_matrix.at[key, index] == 1 else 0))
            num_10 += ((1 if value == 1 else 0) and (1 if ass_matrix.at[key, index] == 0 else 0))
            num_11 += ((1 if value == 1 else 0) and (1 if ass_matrix.at[key, index] == 1 else 0))
            denom_0 += 1 if value == 0 else 0
            denom_1 += 1 if value == 1 else 0

        # if denom_0 > 0 :
        validity.at[index, "q00"] = num_00 / denom_0;
        validity.at[index, "q01"] = num_01 / denom_0
        # if denom_1 > 0:
        validity.at[index, "q10"] = num_10 / denom_1;
        validity.at[index, "q11"] = num_11 / denom_1
    return validity, pseudo


"@ function to find the lowest reliability score"
def get_lowest_reliable_course(reliability):
    max = -math.inf
    course_id = 0
    for index,rows in reliability.iterrows():
        val = -(rows["pi0"]*math.log(rows["pi0"])  + rows["pi1"]*math.log(rows["pi1"]))
        if val > max :
            max = val
            course_id = index
    return course_id


"@ function to generate questions"
def generate_question(ass_data):
    course_id =''
    for index,val in ass_data.get_pesudo().items():
        if(val == 1 and  gb_rev_nodes[int(index)] not in ass_data.used_courses):
            course_id = gb_rev_nodes[int(index)]
            ass_data.courses += [course_id]
            break
    print(course_id)
    know_list = gb_kc_rel[course_id]
    questions = {}
    count =0
    while len(know_list) > 0 and count < 3:
        count += 1
        knowledge = know_list[random.randint(0,len(know_list)-1)]
        knowledge = "Know_235"
        question = Question.query.filter_by(know=knowledge).first()
        questions[str(question.get_question())+str(count)] = str(question.get_answer())
        print("Question")
    ass_data.set_question(questions)
    ass_data.set_course_id(course_id)
    ass_data.set_course_name(gb_courses[course_id])
    ass_data.set_count(ass_data.get_count() + 1)
    print("Generated question")

"@ function to update reliability based on user assessment"
def update_reliability(ass_data):
    qids = ass_data.get_qids()
    qid = qids[len(qids) - 1]
    question =  Question.query.filter_by(id=qid).first()
    user = User.query.filter_by(id=ass_data.get_user_id()).first()
    ti = 0 if ass_data.response == question.get_answer() else 1
    correct = True if ti == 0 else False

    "Dummy ti which should be removed later."
    ti = 1 if ass_data.response == "Yes" else 0

    response = Response(user.get_id(),ass_data.get_course_id(),question.get_id(),ass_data.response,correct)
    response.user1 = user
    response.ques = question
    db.session.add(response)
    db.session.commit()
    reliability = ass_data.get_reliability()
    if ti == 0 :
        reliability.at[ass_data.get_course_id(),"pi0"] = 1
        reliability.at[ass_data.get_course_id(), "pi1"] = 0
    else:
        reliability.at[ass_data.get_course_id(), "pi0"] = 0
        reliability.at[ass_data.get_course_id(), "pi1"] = 1
    ass_data.set_reliability(reliability)


" @function to run em algorithm"
def run_em(reliability, validity, ass_matrix):
    """
    Expectation - step
    """
    w0 = ((reliability.loc[:,"pi0"]).sum())/(reliability.shape[0])
    w1 = ((reliability.loc[:,"pi1"]).sum())/(reliability.shape[0])
    for course,rows in reliability.iterrows():
        num0 = w0;num1 = w1;
        for skill , rows1 in validity.iterrows():
             pow0 = 1 if ass_matrix.loc[course,skill] == 0 else 0
             pow1 = 1 if ass_matrix.loc[course,skill] == 1 else 0
             zero= (math.pow(rows1["q00"],pow0))*(math.pow(rows1["q01"],pow1))
             one = (math.pow(rows1["q10"],pow0))*(math.pow(rows1["q11"],pow1))
             num0 *= zero;num1 *= one;
        reliability.at[course,"pi0"] = num0/(num0 + num1)
        reliability.at[course,"pi1"] = num1/(num0+ num1)

    """
    Maximization - step
    """
    for skill, rows in validity.iterrows():
        num00 = 0; num01 = 0; num10 =0;num11 = 0;
        for course,rows1 in reliability.iterrows():
            num00 += reliability.loc[course,"pi0"]*(1 if ass_matrix.loc[course,skill] == 0 else 0)
            num01 += reliability.loc[course,"pi0"]*(1 if ass_matrix.loc[course,skill] == 1 else 0)
            num10 += reliability.loc[course,"pi1"]*(1 if ass_matrix.loc[course,skill] == 0 else 0)
            num11 += reliability.loc[course,"pi1"]*(1 if ass_matrix.loc[course,skill] == 1 else 0)

        q00 = num00/(num00 + num01);q01 = num01/(num00 + num01);
        q10 = num10/(num10 + num11);q11 = num11/(num10 + num11);
        validity.at[skill,"q00"] = q00;
        validity.at[skill,"q01"] = q01;
        validity.at[skill,"q10"] = q10;
        validity.at[skill,"q11"] = q11;

    return reliability,validity

" @ function to get course relative importance"
def get_relative_importance(skill_list):
    n = len(gb_courses);rows = [];cols = [];
    for course in gb_courses.keys():
        rows += [int(gb_nodes[course])]
    for skill in  skill_list:
        cols += [gb_nodes[gb_skill[skill]]]
    data = np.full((n, 1), 0, dtype=int)
    rel_imp_matrix = pd.DataFrame(data,
                                  index=pd.Index(rows, name="Courses"),
                                  columns=pd.Index([1], name="Score"))
    "Course Relative Importance"
    centroid = [0] * len(gb_node_emb)
    copy_node_emb = copy.deepcopy(gb_node_emb)
    copy_node_emb['centroid'] = centroid
    new_cols = [int(col) for col in cols if int(col) in copy_node_emb.columns]
    copy_node_emb['centroid'] = copy_node_emb[new_cols].sum(axis=1)
    copy_node_emb['centroid'] = copy_node_emb['centroid'].div(len(new_cols))
    denom = 0
    for row in rows:
        if int(row) in copy_node_emb.columns:
            denom += (cosine_similarity(
                copy_node_emb[int(row)].values.reshape(1, -1),
                copy_node_emb['centroid'].values.reshape(1, -1)))[0][0]
    for row in rows:
        numer = 0
        if int(row) in copy_node_emb.columns:
            numer = (cosine_similarity(
                copy_node_emb[int(row)].values.reshape(1, -1),
                copy_node_emb['centroid'].values.reshape(1, -1)))[0][0]
            if denom != 0:
                rel_imp_matrix.loc[row, 1] = numer / denom

    rel_imp_matrix = rel_imp_matrix.sort_values(by=1, axis=0, ascending=False, inplace=False, kind='quicksort')
    return rel_imp_matrix


"@function for generating pseudo recommendation"
def generate_pseudo_recomm(ass_matrix,category,pseudo):
    ass_matrix['sum'] = ass_matrix[list(ass_matrix.columns)].sum(axis=1)
    ass_matrix = ass_matrix.sort_values(by='sum', axis=0, ascending=False, inplace=False, kind='quicksort')
    count = 0
    for index, rows in ass_matrix.iterrows():
        if index not in pseudo.keys() and index in gb_rev_nodes.keys() and gb_rev_nodes[index] in gb_courses.keys() and gb_rev_nodes[index]  in  gb_courses_desc.keys() and (count < 10):
            pseudo[index] = { "courseid" : gb_rev_nodes[index] , "coursename" : gb_courses[gb_rev_nodes[index]]  ,"coursedesc" : gb_courses_desc[gb_rev_nodes[index]] , "response" : "K" , "category": category,"rank" : count}
            count += 1
    return pseudo


"@ function for starting assesment test"
def start_assesment(data):
    pseudo_recom = {}
    ass_data = build_assesment_data(data)
    path = "./crs/resources/data/"
    ass_matrix = pd.read_csv(os.path.join(path, gb_ass_matrix[ass_data.get_category1()]))
    generate_pseudo_recomm(ass_matrix,ass_data.get_category1(),pseudo_recom)
    ass_matrix = pd.read_csv(os.path.join(path, gb_ass_matrix[ass_data.get_category2()]))
    generate_pseudo_recomm(ass_matrix, ass_data.get_category2(), pseudo_recom)
    ass_matrix = pd.read_csv(os.path.join(path, gb_ass_matrix[ass_data.get_category1()]))
    generate_pseudo_recomm(ass_matrix, ass_data.get_category3(), pseudo_recom)
    ass_data.set_pseudo(pseudo_recom)

    return ass_data


"@ function for next question generation"
def next_question(data):
    ass_data = parse_assesment_data(data)
    recommend = 0
    if ass_data.response != "Yes":
        recommend = 1
    response = Response(ass_data.get_user_id(),ass_data.get_course_id(),ass_data.get_course_name(),ass_data.get_response(),recommend)
    db.session.add(response)
    db.session.commit()
    ass_data.used_courses += [ass_data.course_id]
    if(recommend == 1):
        ass_data.recommend += [ass_data.course_id]
    generate_question(ass_data)
    return ass_data

"@ function for user satisfaction"
def user_satisfaction(data):
    ass_data = parse_assesment_data(data)
    refine_recommend = {}
    not_recommend = {}
    gt ={}
    for key,value in ass_data.pseudo.items():
        if value["response"] == "K":
            response = Response(ass_data.get_user_id(), value["courseid"], value["coursename"],
                                value["response"],0,value["category"])
            db.session.add(response)
            db.session.commit()
            not_recommend[key] = value
        else:
            response = Response(ass_data.get_user_id(), value["courseid"], value["coursename"],
                                value["response"], 1, value["category"])
            db.session.add(response)
            db.session.commit()
            refine_recommend[key] = value
        gt[key] = value
        gt[key]["choice"] =  "SA"

    gt = dict(sorted(gt.items(), key=lambda kv: kv[1]["rank"]))
    not_recommend = dict(sorted(not_recommend.items(), key=lambda kv: kv[1]["rank"]))
    refine_recommend = dict(sorted(refine_recommend.items(), key=lambda kv: kv[1]["rank"]))
    sat_data = UserSatisfaction(ass_data.user_id)
    sat_data.set_gt(gt)
    sat_data.set_not_recommend(not_recommend)
    sat_data.set_pseudo_recommend(ass_data.get_pesudo())
    sat_data.set_refine_recommend(refine_recommend)
    sat_data.set_count(ass_data.get_count())
    sat_data.set_category1(ass_data.get_category1())
    sat_data.set_category2(ass_data.get_category2())
    sat_data.set_category3(ass_data.get_category3())
    sat_data.set_category(ass_data.get_category1())
    return sat_data

"@ function for parsing user satisfaction data"
def parse_user_sat_data(data):
    sat_data = UserSatisfaction(data.get("user_id", None))
    sat_data.set_refine_quality(data.get("refine_quality", "No"))
    sat_data.set_category(data.get("category", "NA"))
    sat_data.set_category1(data.get("category1", "NA"))
    sat_data.set_category2(data.get("category2", "NA"))
    sat_data.set_category3(data.get("category3", "NA"))
    sat_data.set_count(data.get("count", 0))
    sat_data.set_pseudo_recommend(data.get("pseudo_recommend", {}))
    sat_data.set_refine_recommend(data.get("refine_recommend", {}))
    sat_data.set_not_recommend(data.get("not_recommend", {}))
    sat_data.set_gt(data.get("gt", {}))
    return sat_data

"@ function for user satisfaction"
def recommend_courses(data):
    sat_data = parse_user_sat_data(data)
    db.session.add(sat_data)
    db.session.commit()
    return sat_data

"@ function for generating report"
def generate_report(data):
    sat_data = parse_user_sat_data(data)
    path = os.path.join('/home/nakopa/crs-project/crs/results/', str(sat_data.get_user_id()))
    if sat_data.count == 1:
        os.mkdir(path)
    path = os.path.join(path, str(sat_data.count))
    os.mkdir(path)

    " Storing ground truth values "
    gt = sat_data.get_gt()
    with open(os.path.join(path,"ground_truth.csv"), 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        rows = []
        rows.append(["UserID", "CourseID", "Course Name",  "Choice" ,"Ground Truth"])
        for key, value in gt.items():
            if value["category"] == sat_data.get_category():
                csv_row = []
                csv_row.append(sat_data.get_user_id())
                csv_row.append(value["courseid"])
                csv_row.append(value["coursename"])
                csv_row.append(value["choice"])
                gt_value = 0
                if str(gb_nodes[value["courseid"]]) in sat_data.get_refine_recommend().keys():
                    if(value["choice"] == "SA" ):
                        gt_value = 2
                    elif (value["choice"] == "A" ):
                        gt_value = 1
                    elif (value["choice"] == "D"):
                        gt_value = -1
                    elif (value["choice"] == "SD"):
                        gt_value = -2
                    else :
                        gt_value = 0
                if str(gb_nodes[value["courseid"]]) in sat_data.get_not_recommend().keys():
                    if (value["choice"] == "SA"):
                        gt_value = 2
                    elif (value["choice"] == "A"):
                        gt_value = 1
                    elif (value["choice"] == "D"):
                        gt_value = -1
                    elif (value["choice"] == "SD"):
                        gt_value = -2
                    else:
                        gt_value = 0
                csv_row.append(gt_value)
                rows.append(csv_row)
                gt_obj = GroundTruth(sat_data.get_user_id(),value["courseid"],value["coursename"],value["choice"],gt_value)
                db.session.add(gt_obj)
                db.session.commit()
        writer.writerows(rows)
        csv_file.close()
    " Storing user profile "
    if sat_data.count == 1 :
        saved_user = User.query.filter_by(id=sat_data.get_user_id()).first()
        job = Job.query.filter_by(user_id=saved_user.get_id()).first()
        with open(os.path.join(path, "user_profile.csv"), 'w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            rows = []
            rows.append(["UserID", "Username", "Job Category 1", "Job Category 2", "Job Category 3"])
            csv_row = []
            csv_row.append(saved_user.get_id())
            csv_row.append(saved_user.get_username())
            csv_row.append(job.get_category1())
            csv_row.append(job.get_category2())
            csv_row.append(job.get_category3())
            rows.append(csv_row)
            writer.writerows(rows)
            csv_file.close()

    " Storing user assessment responses "
    if sat_data.count == 1 :
        with open(os.path.join(path, "user_assessment.csv"), 'w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            rows = []
            rows.append(["UserID", "CourseID", "Course Name", "Response", "Recommendation","Category"])
            response_list = Response.query.filter_by(user_id = sat_data.get_user_id()).all()
            print(len(response_list))
            for response in response_list:
                csv_row = []
                csv_row.append(response.user_id)
                csv_row.append(response.course_id)
                csv_row.append(response.course_name)
                csv_row.append(response.response)
                csv_row.append(response.recommend)
                csv_row.append(response.category)
                rows.append(csv_row)
            writer.writerows(rows)
            csv_file.close()

    " Storing user satisfaction responses "
    if sat_data.count == 3 :
        with open(os.path.join(path, "user_satisfaction.csv"), 'w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            rows = []
            rows.append(["UserID", "Refine Quality", "Category"])
            sat_obj_list = UserSatisfaction.query.filter_by(user_id=sat_data.get_user_id()).all()
            for sat_obj in sat_obj_list :
                csv_row = []
                csv_row.append(sat_obj.user_id)
                csv_row.append(sat_obj.refine_quality)
                csv_row.append(sat_obj.category)
                rows.append(csv_row)
            writer.writerows(rows)
            csv_file.close()

    " Storing refined recommendation "
    with open(os.path.join(path, "actual_recommendation.csv"), 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        rows = []
        rows.append(["CourseID", "CourseName", "Category" ,"Recommendation"])
        for key, value in sat_data.get_refine_recommend().items():
            if value["category"] == sat_data.category :
                csv_row = []
                csv_row.append(value["courseid"])
                csv_row.append(value["coursename"])
                csv_row.append(value["category"])
                csv_row.append(1)
                rows.append(csv_row)
        writer.writerows(rows)
        csv_file.close()

    " Storing not recommended courses"
    with open(os.path.join(path, "not_recommended.csv"), 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        rows = []
        rows.append(["CourseID", "CourseName", "Category" ,"Recommendation" ])
        for key, value in sat_data.get_not_recommend().items():
            if value["category"] == sat_data.category:
                csv_row = []
                csv_row.append(value["courseid"])
                csv_row.append(value["coursename"])
                csv_row.append(value["category"])
                csv_row.append(0)
                rows.append(csv_row)
        writer.writerows(rows)
        csv_file.close()

    sat_data.set_count(sat_data.get_count() + 1)
    if(sat_data.count == 1):
        sat_data.set_category(sat_data.get_category1())
    elif sat_data.count == 2 :
        sat_data.set_category(sat_data.get_category2())
    else :
        sat_data.set_category(sat_data.get_category3())
    sat_data.set_refine_quality("")
    return sat_data

