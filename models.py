from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import json

db = SQLAlchemy()

class SummarySession(db.Model):
    __tablename__ = 'summary_sessions'
    
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    original_text = db.Column(db.Text, nullable=False)
    summary_text = db.Column(db.Text, nullable=False)
    level = db.Column(db.String(20), nullable=False)
    word_count = db.Column(db.Integer)
    input_type = db.Column(db.String(20))
    source_info = db.Column(db.String(500))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    questions = db.relationship('QuizQuestion', backref='session', lazy=True, cascade='all, delete-orphan')
    
    def to_dict(self):
        return {
            'id': self.id,
            'title': self.title,
            'summary_text': self.summary_text,
            'level': self.level,
            'word_count': self.word_count,
            'input_type': self.input_type,
            'source_info': self.source_info,
            'created_at': self.created_at.isoformat(),
            'questions': [q.to_dict() for q in self.questions]
        }

class QuizQuestion(db.Model):
    __tablename__ = 'quiz_questions'
    
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey('summary_sessions.id'), nullable=False)
    question_text = db.Column(db.Text, nullable=False)
    correct_answer = db.Column(db.Text, nullable=False)
    user_answer = db.Column(db.Text)
    is_correct = db.Column(db.Boolean)
    feedback = db.Column(db.Text)
    question_order = db.Column(db.Integer)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'question': self.question_text,
            'answer': self.correct_answer,
            'user_answer': self.user_answer,
            'is_correct': self.is_correct,
            'feedback': self.feedback,
            'question_order': self.question_order
        }