"""empty message

Revision ID: 35291f715aa0
Revises: 
Create Date: 2021-01-20 01:31:23.244047

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '35291f715aa0'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('question',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('ques_id', sa.String(length=255), nullable=True),
    sa.Column('question', sa.String(length=255000), nullable=True),
    sa.Column('answer', sa.String(length=255000), nullable=True),
    sa.Column('know', sa.String(length=255000), nullable=True),
    sa.Column('level', sa.String(length=255000), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('user',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('username', sa.String(length=255), nullable=True),
    sa.Column('email', sa.String(length=255), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('ground_truth',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('user_id', sa.Integer(), nullable=True),
    sa.Column('course_id', sa.String(length=255), nullable=True),
    sa.Column('course_name', sa.String(length=255), nullable=True),
    sa.Column('choice', sa.String(length=255), nullable=True),
    sa.Column('gt', sa.Integer(), nullable=True),
    sa.ForeignKeyConstraint(['user_id'], ['user.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('job',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('user_id', sa.Integer(), nullable=True),
    sa.Column('category1', sa.String(length=80), nullable=True),
    sa.Column('category2', sa.String(length=80), nullable=True),
    sa.Column('category3', sa.String(length=80), nullable=True),
    sa.ForeignKeyConstraint(['user_id'], ['user.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('response',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('user_id', sa.Integer(), nullable=True),
    sa.Column('course_id', sa.String(length=255), nullable=True),
    sa.Column('course_name', sa.String(length=255), nullable=True),
    sa.Column('response', sa.String(length=255), nullable=True),
    sa.Column('recommend', sa.Integer(), nullable=True),
    sa.Column('category', sa.String(length=255), nullable=True),
    sa.ForeignKeyConstraint(['user_id'], ['user.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('user_satisfaction',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('user_id', sa.Integer(), nullable=True),
    sa.Column('refine_quality', sa.String(length=255), nullable=True),
    sa.Column('category', sa.String(length=255), nullable=True),
    sa.ForeignKeyConstraint(['user_id'], ['user.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table('user_satisfaction')
    op.drop_table('response')
    op.drop_table('job')
    op.drop_table('ground_truth')
    op.drop_table('user')
    op.drop_table('question')
    # ### end Alembic commands ###
