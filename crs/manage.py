
from flask_sqlalchemy import SQLAlchemy



db = SQLAlchemy()


def data_init():
    print('data init')


if __name__ == '__main__':
    data_init()