import os
from app import app
import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename

ALLOWED_EXTENSIONS = set(['wav'])

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS #check the extension of file
	
@app.route('/')
def upload_form():
	return render_template('upload.html')

@app.route('/', methods=['POST'])
def upload_audio():
	if 'file' not in request.files: #to check if file uploaded correctly
		flash('No file part')
		return redirect(request.url)
	file = request.files['file']
	if file.filename == '':
		flash('No image selected for uploading')
		return redirect(request.url)
	if file and allowed_file(file.filename):
		filename = secure_filename(file.filename) #the filename should be correctly written no spaces or special charactor
		#file.filename = '1.jpg'
		file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))

		import audio_flask_classification #importing the classifier on file
		out, dic = audio_flask_classification.main(filename) #class prediction
		desc = dic['Description'].item()
		english_cname = dic['english_cname'].item()
		species = dic['species'].item()
		genus = dic['genus'].item()
		family = dic['Family'].item()
		link = dic['Link'].item()
		imag = dic['Img'].item()
		flash('Audio successfully uploaded and displayed')
		return render_template('upload.html', filename=file.filename, result = english_cname, desc = desc, species = species, family = family, link = link, genus = genus, imag = imag)
	else:
		flash('Allowed image type is ->  wav')
		return redirect(request.url)

@app.route('/display/<filename>')
def display_audio(filename):
	return redirect(url_for('static', filename='uploads/'+filename))

if __name__ == "__main__":
    app.run()
