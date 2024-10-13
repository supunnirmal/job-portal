

import docx2txt
 
# read in word file
result = docx2txt.process("C:/Users/Supun/Desktop/New Microsoft Word Document.docx")

#select area
start = 'Produce and test prototype implementations of systems to help answer research questions.'
end = 'Database design and programming'
list_word = result[result.find(start)+len(start):result.rfind(end)]
#print(list_word)

#extract emails
import re

my_str = result
emails = re.findall("([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)", my_str)

for mail in emails:
    print(mail)

#extract phone numbers

numbers = re.findall(r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]', result)
for number in numbers:
    print(number)

import pandas as pd
import en_core_web_sm
nlp = en_core_web_sm.load()
import spacy
from spacy.matcher import PhraseMatcher
from collections import Counter

import os
from os import listdir
from os.path import isfile, join
from io import StringIO


import csv


#text = pdfextract(file) 
text = str(result)
text = text.replace("\\n", "")
text = text.lower()
#print(text)
#below is the csv where we have all the keywords, you can customize your own
keyword_dict = pd.read_csv('C:/Users/Supun/Desktop/Group project/portal/Key_Words/Book1.csv')

SWD_words = [nlp(text) for text in keyword_dict['Software_Development'].dropna(axis = 0)]

#for words in SWD_words:
#    print(words)

matcher = PhraseMatcher(nlp.vocab)
matcher.add('SWD', None, *SWD_words)
doc = nlp(text)
    
d = []  
matches = matcher(doc)
for match_id, start, end in matches:
    rule_id = nlp.vocab.strings[match_id]  # get the unicode ID, i.e. 'COLOR'
    span = doc[start : end]  # get the matched slice of the doc
    d.append((rule_id, span.text))      
    
keywords = "\n".join(f'{i[0]} {i[1]} ({j})' for i,j in Counter(d).items())
    
    ## convertimg string of keywords to dataframe
df = pd.read_csv(StringIO(keywords),names = ['Keywords_List'])
df1 = pd.DataFrame(df.Keywords_List.str.split(' ',1).tolist(),columns = ['Subject','Keyword'])
df2 = pd.DataFrame(df1.Keyword.str.split('(',1).tolist(),columns = ['Keyword', 'Count'])
df3 = pd.concat([df1['Subject'],df2['Keyword'], df2['Count']], axis =1) 
df3['Count'] = df3['Count'].apply(lambda x: x.rstrip(")"))
    
base = os.path.basename(result)
filename = os.path.splitext(base)[0]
       
name = filename.split('_')
name2 = name[0]
name2 = name2.lower()
    ## converting str to dataframe
name3 = pd.read_csv(StringIO(name2),names = ['Candidate Name'])
    
dataf = pd.concat([name3['Candidate Name'], df3['Subject'], df3['Keyword'], df3['Count']], axis = 1)
dataf['Candidate Name'].fillna(dataf['Candidate Name'].iloc[0], inplace = True)


        
#function ends
        
#code to execute/call the above functions

final_database=pd.DataFrame()
i = 0 
#while i < len(onlyfiles):
#    file = onlyfiles[i]
    #dat = create_profile(file)

final_database = final_database.append(dataf)
print(final_database)


#code to count words under each category and visulaize it through Matplotlib

final_database2 = final_database['Keyword'].groupby([final_database['Candidate Name'], final_database['Subject']]).count().unstack()
final_database2.reset_index(inplace = True)
final_database2.fillna(0,inplace=True)
new_data = final_database2.iloc[:,1:]
new_data.index = final_database2['Candidate Name']
#execute the below line if you want to see the candidate profile in a csv format
#sample2=new_data.to_csv('sample.csv')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 10})
ax = new_data.plot.barh(title="Resume keywords by category", legend=False, figsize=(25,7), stacked=True)
labels = []
for j in new_data.columns:
    for i in new_data.index:
        label = str(j)+": " + str(new_data.loc[i][j])
        labels.append(label)
patches = ax.patches
for label, rect in zip(labels, patches):
    width = rect.get_width()
    if width > 0:
        x = rect.get_x()
        y = rect.get_y()
        height = rect.get_height()
        ax.text(x + width/2., y + height/2., label, ha='center', va='center')
plt.show()