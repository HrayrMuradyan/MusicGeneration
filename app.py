from flask import Flask, render_template, request, redirect, url_for, send_file
from flask_wtf import FlaskForm
from wtforms import TextAreaField, SubmitField, validators
import os
import uuid
import json
import shutil
from MusicGenAI import MusicGenAI
from scipy.io import wavfile
import argparse

app = Flask(__name__)
app.config['SECRET_KEY'] = 'df64008f89557cbc33db9a5a70291af2'

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', help='Path to the model checkpoint')

class TextForm(FlaskForm):
    user_text = TextAreaField('Provide a description for the music you want to generate. You can specify Instruments, mood and genre',
                              [validators.Length(min=1, max=500, message='Text must be between 1 and 500 characters.')],
                              render_kw={"placeholder": "Enter your text here..."})
    submit = SubmitField('Generate Audio')


@app.route('/', methods=['GET', 'POST'])
def index():
    form = TextForm()
    audio_file_path = None

    if form.validate_on_submit():
        # Get the user's input text
        user_text = form.user_text.data

        # Call the generate function (here we use gTTS)
        audio_file_path = generate_audio(user_text)

        # Redirect to allow play/download the audio and show feedback form
        return redirect(url_for('audio', file_path=audio_file_path))

    # Render the form
    return render_template('index.html', form=form)


@app.route('/audio')
def audio():
    # Retrieve the file path from query parameter
    file_path = request.args.get('file_path')

    # Check if file exists
    if not os.path.exists(file_path):
        return "Audio file not found."

    # Render the audio playback and download page
    return render_template('audio.html', audio_file_path=file_path)


@app.route('/download/<path:file_path>')
def download(file_path):
    # Send the file as an attachment for download
    return send_file(file_path, as_attachment=True)


def generate_audio(text):
    # Generate a unique filename
    filename = f"{uuid.uuid4()}.mp3"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    # shutil.copyfile('download.wav', filepath)

    music = model.generate_music(text)
    print(music.shape)
    wavfile.write(filepath, 32000, music[0]) 
    
    return filepath


@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    # Retrieve rating from the request form data
    rating = request.form.get('rating')
    comments = request.form.get('comments', '')

    # Save feedback to a JSON file (or database)
    feedback = {
        'rating': rating,
        'comments': comments
    }

    feedback_file = 'feedback.json'
    # Load existing feedback data
    if os.path.exists(feedback_file):
        with open(feedback_file, 'r') as file:
            data = json.load(file)
    else:
        data = []

    # Append new feedback data
    data.append(feedback)

    # Save feedback data to the file
    with open(feedback_file, 'w') as file:
        json.dump(data, file, indent=4)

    # Redirect back to the homepage
    return redirect(url_for('index'))


if __name__ == '__main__':
    # Configure the file upload path
    UPLOAD_FOLDER = 'static/audio'
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

    args = parser.parse_args()

    # Define a global variable
    model = MusicGenAI()

    # Create an upload folder if it doesn't exist
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    
    
    with app.app_context():
        # Initialize model
        model.load_model(args.checkpoint)

    app.run(debug=False)
