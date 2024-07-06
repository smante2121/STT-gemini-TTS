from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired

class LookUpForm(FlaskForm):
    number = StringField('Phone Number', validators=[DataRequired()])
    submit = SubmitField('Submit')