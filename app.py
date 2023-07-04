from flask import Flask, render_template, redirect, url_for, request, flash
from werkzeug.utils import secure_filename
from uploads.file_handler import is_file_type_allowed, upload_file_to_s3, get_presigned_file_url
from localStoragePy import localStoragePy
from transformers import AutoTokenizer, pipeline
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.models import load_model
from taipy.gui import Gui

import webbrowser
import tensorflow as tf
import pandas as pd
import numpy as np
import pytorch_pretrained_bert as ppb
assert 'bert-large-cased' in ppb.modeling.PRETRAINED_MODEL_ARCHIVE_MAP
app = Flask(__name__)
app.secret_key = '3d6f45a5fc12445dbac2f59c3b6c7cb1'
localStorage = localStoragePy('app', 'json')
target_arr = ["df['col1'].nunique()",
             "df.sort_values(by=['col1'],inplace =True)",
             "df.sort_values(by=['col1', 'col2'],inplace =True)",
             "df.sort_values(by=['col1', 'col2', 'col3'],inplace =True)",
             "df.drop(columns = 'col1',inplace = True)",
             "new_df=df.loc[:, ['col1','col2']]",
             "df['col1'].value_counts()",
             "<|{dataset}|chart|type=bar|x=col1|y=col2|height=100%|>",
             "<|{dataset}|chart|type=pie|values=col2|labels=col1|height=100%|>",
             "<|{dataset}|chart|mode=lines|x=col1|y=col2|>"]
portNo = 8888
@app.route("/", methods=['GET'])
def home():    
    return render_template('home.html')

@app.route("/upload-file", methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file uploaded', 'danger')
        return redirect(url_for('home'))
    
    file_to_upload = request.files['file']
    if file_to_upload.filename == '':
        flash('No file uploaded', 'danger')
        return redirect(url_for('home'))
    
    if file_to_upload and is_file_type_allowed(file_to_upload.filename):
        provided_file_name = secure_filename(file_to_upload.filename)
        stored_file_name = upload_file_to_s3(file_to_upload, provided_file_name)

        localStorage.setItem("stored_file_name", stored_file_name)
        localStorage.setItem("provided_file_name", provided_file_name)

        flash(f'{provided_file_name} was successfully uploaded', 'success')
    
    return redirect(url_for('home'))

@app.route("/query", methods=['POST'])
def query():
    try:
        query = request.form['query']
        provided_file_name = localStorage.getItem("provided_file_name")
        stored_file_name = localStorage.getItem("stored_file_name")
        csv = get_presigned_file_url(stored_file_name, provided_file_name)
        df = pd.read_csv(csv)
        print("query: " + query)
        prediction_int, cols_requested = getPredictionInt(df, query)
        if prediction_int < 7:
            panda_query = target_arr[prediction_int]
            print(panda_query)
            for i in range(len(cols_requested)):
                panda_query = panda_query.replace("col" + str(i+1), cols_requested[i])
            exec(panda_query)
            html_string = '''
                <html lang="en">
                <head>
                    <meta charset="UTF-8">
                    <meta http-equiv="X-UA-Compatible" content="IE=edge">
                    <meta name="viewport" content="width=device-width, initial-scale=1.0">
                    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.0.2/dist/css/bootstrap.min.css" rel="stylesheet">
                </head>
                    <body>
                        <div class="justify-content-center mt-5">
                            <div class="text-center">
                                <h4 class="">Download CSV 
                                    <a href='{new_presigned_url}'>
                                    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" style="width:20px; height:20px;">
                                        <path stroke-linecap="round" stroke-linejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5M16.5 12L12 16.5m0 0L7.5 12m4.5 4.5V3" />
                                    </svg>
                                    </a>
                                </h4>
                                <p>Click <a href='/'>here</a> to return to home page</p>
                                {table}
                            </div>
                        </div>
                    </body>
                </html>.
            '''
            filename = "new.csv"
            df.to_csv(filename, index=False)
            file_to_upload = open("new.csv", 'rb')
            new_provided_file_name = secure_filename(filename)
            new_stored_file_name = upload_file_to_s3(file_to_upload, new_provided_file_name)
            
            new_presigned_url = get_presigned_file_url(new_stored_file_name, new_provided_file_name)
            print("Presigned url: " + new_presigned_url)
            df = df.reset_index(drop=True)
            html = df.to_html(classes='table table-striped table-bordered w-75 mx-auto')
            html = html.replace("text-align: right;", "text-align: left;")
            toDisplay = html_string.format(table = html, new_presigned_url = new_presigned_url)
            return toDisplay
        else:
            print("taipy")
            taipy_query = target_arr[prediction_int]
            dataset = df
            for i in range(len(cols_requested)):
                taipy_query = taipy_query.replace("col" + str(i+1), cols_requested[i])
            page = """{0}"""
            page = page.format(taipy_query)
            gui = Gui(page)
            global portNo
            portNum = portNo
            portNo += 1
            webbrowser.open_new_tab('http://localhost:' + str(portNum))
            gui.run(port=portNum)
            print("hello world")
            return redirect(url_for('home'))
    except:
        print("Invalid query")
        flash('Invalid query', 'danger')
        return redirect(url_for('home'))
def getPredictionInt(df, query):
    
    cols = df.columns
    sentence = query
    words = sentence.split()
    cols_requested = []
    for item in cols:
        for word in words:
            if(item.upper() == word.upper()):
                cols_requested.append(item)

    general_sentence = sentence
    for i in range(len(cols_requested)):
        general_sentence = general_sentence.replace(cols_requested[i], "col" + str(i+1))

    model_id = "tanishabhagwanani/distilbert-base-uncased-finetuned-emotion"
    classifier = pipeline("text-classification", model=model_id)
    custom_question = query
    preds = classifier(custom_question, return_all_scores=True)
    preds_df = pd.DataFrame(preds[0])
    prediction_int = np.argmax(preds_df.score)
    return prediction_int, cols_requested

if __name__=='__main__':
    app.run(host="localhost", port=8000, debug=True)