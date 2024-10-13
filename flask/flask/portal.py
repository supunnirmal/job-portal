from flask import Flask, render_template, request, redirect, url_for
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import docx2txt
import mydict 
import numpy as np
import gensim
from werkzeug import secure_filename
import os
import pymongo
import pandas as pd
import en_core_web_sm
nlp = en_core_web_sm.load()
from spacy.matcher import PhraseMatcher
import re
from nltk.corpus import stopwords


myclient = pymongo.MongoClient("mongodb://localhost:27017/")

mydb = myclient["cv"]
mycol = mydb["Institute"]
mycol1 = mydb["provider"]
comments=[]
jobs=[]
vacancies=[]
wordsmis=[]
qualifications=[]


portal = Flask(__name__)
@portal.route('/')
@portal.route('/home', methods=['GET','POST'])

def index():
    results = []
    for obj in mycol1.find():
        results.append(obj)
    return render_template("index.html",results=results)
   
@portal.route('/vacancy', methods=['GET','POST'])
def vacancy():
    return render_template("job.html")
    
@portal.route('/cv', methods=['GET','POST'])
def cv():
    return render_template("cv.html")
    
@portal.route('/display', methods=['GET','POST'])
def display():
    if request.method == "POST":
        cname = str(request.get_data())
        cname = (cname.split('\"')[1])
        results=[]
        for obj in mycol1.find():
            if(obj['company']==cname):
                results.append(obj)
    return render_template("display.html",results=results)

   #to clear arrays
@portal.route('/clear', methods=['GET','POST'])
def clear():
    global qualifications
    qualifications.clear()
    global wordsmis
    wordsmis.clear()
    global comments
    comments.clear()
    global jobs
    global vacancies
    jobs.clear()
    vacancies.clear()
    return render_template("result.html")

#get vacancies
@portal.route('/getefile', methods=['GET','POST'])
def getefile():
    if request.method == 'POST':
        
        cname = request.form['cname']
        title = request.form['eid']
        location = request.form['location']
        gender = request.form['gender']
        age = request.form['age']
        salary = request.form['salary']
        days = request.form['days']
        hours = request.form['hours']
        role = request.form['role']
        hours = request.form['hours']
        qua = request.form['qua']
        exp = request.form['exp']
        tec = request.form['tec']
        
        
    #function that does phrase matching and builds a candidate profile
    text = str(tec)
    text = text.replace("\\n", "")
    text = text.lower()
    
    keyword_dict = pd.read_csv('Keys.csv')

    NLP_words = [nlp(str(text).lower()) for text in keyword_dict['Keys'].dropna(axis = 0)]

    matcher = PhraseMatcher(nlp.vocab)
    matcher.add('Keys', None, *NLP_words)

    doc = nlp(text)
    
    d = []  
    words=[]
    matches = matcher(doc)
    for match_id, start, end in matches:
        rule_id = nlp.vocab.strings[match_id]  # get the unicode ID, i.e. 'COLOR'
        span = doc[start : end]  # get the matched slice of the doc
        words.append(str(span))
        
    #write_to_csv
    
    with open(cname+'.csv', 'a') as csvfile:
        for domain in words:
            csvfile.write(domain + '\n')

    #write to db
    mydict = {"company" : cname, "job" : title, "location" : location, "gender" : gender, "age" : age, "salary" : salary, "days" : days, "hours" : hours, "role" : role,"hours" : hours, "qualification" : qua, "experence" : exp, "technologies" : tec}
    mycol1.insert_one(mydict)
        
    return redirect(url_for('index'))

#get CV
@portal.route('/getfile', methods=['GET','POST'])
def getfile():
    if request.method == 'POST':
        file = request.files['myfile']
        file.save(os.path.join('cv.docx'))
        
    return render_template("result.html")

 
@portal.route('/text', methods=['GET', 'POST'])


def text():
    global comments
    global jobs
    global vacancies
    global wordsmis
    global qualifications
    
    
    
    if request.method == "GET":
        return render_template("result.html",wordsmis=wordsmis,comments=comments,jobs=jobs,vacancies=vacancies,qualifications=qualifications)    
    
    
    result = docx2txt.process("cv.docx")#read the file
    
    #check minimum quality
    startminq = 'Name of the qualification:'
    endminq = 'Description of the qualification:'
    part = result[result.find(startminq)+len(startminq):result.rfind(endminq)]


    # Grad all general stop words
    STOPWORDS = set(stopwords.words('english'))

    # Education Degrees
    DEGREE = ['BE','B.E.', 'B.E', 'BS', 'B.S','BSC']
    MASTER = ['MSC','ME', 'M.E', 'M.E.', 'MS', 'M.S','M.TECH', 'MTECH']
    PHD =['PHD']

    #def extract_education(resume_text):
    nlp_text = nlp(part)

    # Sentence Tokenizer
    nlp_text = [sent.string.strip() for sent in nlp_text.sents]

    minq =""
    minqfull=""
    # Extract education degree
    for index, text in enumerate(nlp_text):
        for tex in text.split():
            # Replace all special symbols
            tex = re.sub(r'[?|$|.|!|,]', r'', tex)
            if tex.upper() in DEGREE and tex not in STOPWORDS:
                minqfull = text + nlp_text[index + 1]
                minq="degree"
                
            elif tex.upper() in MASTER and tex not in STOPWORDS:
                minqfull = text + nlp_text[index + 1]
                minq="msc"
                
            elif tex.upper() in PHD and tex not in STOPWORDS:
                minqfull = text + nlp_text[index + 1]
                minq="phd"
    
    
    
    
    
    start = 'Education:'
    end = 'Other Skills:'
    list_word = result[result.find(start)+len(start):result.rfind(end)]
    
            
    #tockenize in to sentences
    file_docs = []
    tokens = sent_tokenize(list_word)
    for line in tokens:
        file_docs.append(line)
    

    #tockenize in to words
    gen_docs = [[w.lower() for w in word_tokenize(text)] 
            for text in file_docs]

    #map to unique id
    dictionary = gensim.corpora.Dictionary(gen_docs)

    #create bag of words
    corpus = [dictionary.doc2bow(gen_doc) for gen_doc in gen_docs]
    
    #from gensim import models
    tf_idf = gensim.models.TfidfModel(corpus)
    for doc in tf_idf[corpus]:
        ([[dictionary[id], np.around(freq, decimals=2)] for id, freq in doc])

    # building the index
    sims = gensim.similarities.Similarity("123.docx",tf_idf[corpus],num_features=len(dictionary))

    

    for obj in mycol1.find():
        result2 = str(obj['qualification'])
        if(minq == result2):
            result1 = str(obj['technologies'])
        
            file2_docs = []

            tokens = sent_tokenize(result1)
            for line in tokens:
                file2_docs.append(line)

            for line in file2_docs:
                query_doc = [w.lower() for w in word_tokenize(line)]
                query_doc_bow = dictionary.doc2bow(query_doc)
            
            #update an existing dictionary and create bag of words
            query_doc_tf_idf = tf_idf[query_doc_bow]
    
            sum_of_sims =(np.sum(sims[query_doc_tf_idf], dtype=np.float32))

            avg_sims = [] # array of averages

            # for line in query documents
            for line in file2_docs:
                # tokenize words
                query_doc = [w.lower() for w in word_tokenize(line)]
            
                # create bag of words
                query_doc_bow = dictionary.doc2bow(query_doc)
            
                # find similarity for each document
                query_doc_tf_idf = tf_idf[query_doc_bow]
            
                # calculate sum of similarities for each query doc
                sum_of_sims =(np.sum(sims[query_doc_tf_idf], dtype=np.float32))
            
                # calculate average of similarity for each query doc
                avg = sum_of_sims / len(file_docs)
            
                # add average values into array
                avg_sims.append(avg)  
        
            # calculate total average
            total_avg = np.sum(avg_sims, dtype=np.float)
    
        
            # round the value and multiply by 100 to format it as percentage
            percentage_of_similarity = round(float(total_avg) * 100)
        
            if percentage_of_similarity >= 100:
                percentage_of_similarity = 100
        
        
            obj['id']=str(percentage_of_similarity)+"%"
            comments.append(obj)
        
            if total_avg>0.65:
                jobs.append(obj)
        
            if total_avg<=0.65:
            
                text = str(list_word)
                text = text.replace("\\n", "")
                text = text.lower()
    
                keyword_dict = pd.read_csv(obj['company']+'.csv')

                NLP_words = [nlp(str(text).lower()) for text in keyword_dict['Key-Words'].dropna(axis = 0)]

                matcher = PhraseMatcher(nlp.vocab)
                matcher.add('Key-Words', None, *NLP_words)

                doc = nlp(text)
    
                words=[]
                matches = matcher(doc)
                for match_id, start, end in matches:
                    rule_id = nlp.vocab.strings[match_id]  # get the unicode ID, i.e. 'COLOR'
                    span = doc[start : end]  # get the matched slice of the doc
                    words.append(str(span))
        
                word=[]
                tec="Experience on "
                data = pd.DataFrame(keyword_dict[~keyword_dict.isin(words)])
                for item in (np.asarray(data.dropna())):
                    word.append(str(item))
                    item=str(item)
                    tec+=(re.search(r'\'(.*?)\'',item).group(1)+",\n")
                obj['technologies']=tec
                vacancies.append(obj)
            
            
                with open('missing.csv', 'w') as csvfile:
                    csvfile.write("missing-word"+'\n')
                    for item in word:
                        csvfile.write(re.search(r'\'(.*?)\'',item).group(1) + '\n')
            
            
                for obj in mycol.find():
                    text = str(obj['Content'])
                    text = text.replace("\\n", "")
                    text = text.lower()
                
                    keyword_dict = pd.read_csv('missing.csv')

                    NLP_words = [nlp(str(text).lower()) for text in keyword_dict['missing-word'].dropna(axis = 0)]

                    matcher = PhraseMatcher(nlp.vocab)
                    matcher.add('missing-word', None, *NLP_words)

                    doc = nlp(text)
      
                    words="You can develop "
                    i=0
                    matches = matcher(doc)
                    for match_id, start, end in matches:
                        i=1
                        rule_id = nlp.vocab.strings[match_id]  # get the unicode ID, i.e. 'COLOR'
                        span = doc[start : end]  # get the matched slice of the doc
                        if(str(span) in words):
                            continue
                        else:
                            words+= str(span)+", "
                    if(i==1):        
                        words+="skill(s) or related skill(s)"+"\n"
                        obj['Content']=words
                        wordsmis.append(obj)
                
        else:
            obj['id']=minqfull
            qualifications.append(obj)
    
  
    return redirect(url_for('text'))

 
if __name__ == '__main__':
   portal.run(debug=True)
   
