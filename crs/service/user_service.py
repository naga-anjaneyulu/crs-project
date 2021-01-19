from flask_sqlalchemy import SQLAlchemy

from manage import db
from model.models import User


"@Registers user and returns the saved object. "
def register(username,email):
    user = User(username,email)
    db.session.add(user)
    db.session.commit()
    saved_user_list = User.query.filter_by(username=user.get_username(),email=email).all()
    return saved_user_list[len(saved_user_list) - 1]



"@Login user and returns the saved object. "
def login(username,password):
    saved_user = User.query.filter_by(username=username).first()
    if(saved_user and saved_user.get_password() == password) :
        return saved_user
    else:
        return None