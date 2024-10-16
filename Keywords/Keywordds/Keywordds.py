import PyPDF2 
import textract
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import csv
import re  

#write a for-loop to open many files -- leave a comment if you'd #like to learn how
filename = 'C:/Users/Supun/Desktop/Group project/portal/Jobs/New Microsoft Word Document.pdf' 
#open allows you to read the file
pdfFileObj = open(filename,'rb')
#The pdfReader variable is a readable object that will be parsed
pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
#discerning the number of pages will allow us to parse through all #the pages
num_pages = pdfReader.numPages
count = 0
text = ""
#The while loop will read each page
while count < num_pages:
    pageObj = pdfReader.getPage(count)
    count +=1
    text += pageObj.extractText()
#This if statement exists to check if the above library returned #words. It's done because PyPDF2 cannot read scanned files.
if text != "":
   text = text
#If the above returns as False, we run the OCR library textract to #convert scanned/image based PDF files into text
else:
   text = textract.process(fileurl, method='tesseract', language='eng')
# Now we have a text variable which contains all the text derived #from our PDF file. Type print(text) to see what it contains. It #likely contains a lot of spaces, possibly junk such as '\n' etc.
# Now, we will clean our text variable, and return it as a list of keywords.
text = text.lower()
#text = re.sub('[^a-zA-Z]', ' ', text)
#The word_tokenize() function will break our text phrases into #individual words
tokens = word_tokenize(text)

words = [word for word in tokens if word.isalpha()]

#we'll create a new list which contains punctuation we wish to clean
punctuations = ['(',')',';',':','[',']',',']
#We initialize the stopwords variable which is a list of words like #"The", "I", "and", etc. that don't hold much value as keywords
stop_words = stopwords.words('english')
#We create a list comprehension which only returns a list of words #that are NOT IN stop_words and NOT IN punctuations.
keywords = [word for word in words if not word in stop_words and not word in punctuations]
print(keywords)


#with open('C:/Users/Supun/Desktop/Group project/portal/Key_Words/Book2.txt') as csv_file:
#wtr = csv.writer(open ('C:/Users/Supun/Desktop/Group project/portal/Key_Words/Book1.csv', 'w'), delimiter=',', lineterminator='\n')
#for x in keywords : wtr.writerow ([x])


def write_to_csv(list_of_emails):
    with open('C:/Users/Supun/Desktop/Group project/portal/Key_Words/Book2.csv', 'a') as csvfile:
        for domain in list_of_emails:
            csvfile.write(domain + '\n')

write_to_csv(keywords)