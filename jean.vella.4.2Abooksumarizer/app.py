import tkinter as tk #gui
import PyPDF2 #pdf extraction
from PIL import Image, ImageTk #logo image
from tkinter.filedialog import askopenfile #open pdf and checking it is pdf
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import string
import warnings 
from nltk.corpus import stopwords #for tokenaization stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer #sentiment import
sa = SentimentIntensityAnalyzer()

nltk.download('stopwords') 
HANDICAP = 0.75 #handicap to determine how long summary will be (1 = xej u 0= ma jkunx hemm biddla)
root = tk.Tk()
canvas = tk.Canvas(root, width=1200, height=400) #setting of gui height
canvas.grid(columnspan=3, rowspan=3)

#logo
logo = Image.open('C:/Users/yoshi/Desktop/jean.vella.4.2Abooksumarizer/brand.png') #opening brand image to put
logo = ImageTk.PhotoImage(logo)
logolabel = tk.Label(image=logo)
logolabel.image = logo
logolabel.grid(column=1, row=0) #putting image in the right place

def removepunctuationmarks(text) : #removing unwated punctuations making them tokenized
    punctuationmarks = dict((ord(punctuationmark), None) for punctuationmark in string.punctuation)
    return text.translate(punctuationmarks)

def getlemmatizedtokens(text) : #lemmitazation
    normalizedtokens = nltk.word_tokenize(removepunctuationmarks(text.lower()))
    return [nltk.stem.WordNetLemmatizer().lemmatize(normalizedtoken) for normalizedtoken in normalizedtokens]

def getaverage(values) : #getting the average values to make the summary
    greaterthanzerocount = total = 0
    for value in values :
        if value != 0 :
            greaterthanzerocount += 1
            total += value 
    return total / greaterthanzerocount

def getthreshold(tfidfresults) : #getting the threshold with the average function
    i = total = 0
    while i < (tfidfresults.shape[0]) :
        total += getaverage(tfidfresults[i, :].toarray()[0])
        i += 1
    return total / tfidfresults.shape[0]

def getsummary(documents, tfidfresults) : #to do the actual summary using the handicap
    summary = ""
    i = 0
    while i < (tfidfresults.shape[0]) :
        if (getaverage(tfidfresults[i, :].toarray()[0])) >= getthreshold(tfidfresults) * HANDICAP :
                summary += ' ' + documents[i]
        i += 1
    return summary




instructions = tk.Label(root, text="Select a PDF file on your computer to extract all its text", font="Raleway")
instructions.grid(columnspan=3, column=0, row=1)

def openfile():
    browsetext.set("loading...") #to make the user wait
    file = askopenfile(parent=root, mode='rb', title="Choose a file", filetypes=[("Pdf file", "*.pdf")]) #what the user sees when they press browse and it will give only .pdf files
    if file:
        readpdf = PyPDF2.PdfReader(file)
        
        page = readpdf.pages[0]
        pagecontent = page.extract_text() #if file is chosen extract text
        
        
        if __name__ == "__main__" :
            warnings.filterwarnings("ignore")

            try :
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                print('punkt')
                nltk.download('punkt')

            try :
                nltk.data.find('corpora/wordnet') #if there is errors
            except LookupError:
                nltk.download('wordnet')
            
            documents = nltk.sent_tokenize(pagecontent) #tokenizing page content

            tfidfresults = TfidfVectorizer(tokenizer = getlemmatizedtokens, stop_words = stopwords.words('english')).fit_transform(documents) #lematizing the text from the pdf
            finalsummary = (getsummary(documents, tfidfresults)) # getting the final summary
            sentimenttext = "" 
            score = sa.polarity_scores(finalsummary) #reading the text
            sentimentanalysis = (score['compound']) #getting the sentiment analysis score (inqas min 0 neggativ u iktar min 0 positiv)
            if(sentimentanalysis < -0.6):
                sentimenttext = "Very Negative Book."
            if(sentimentanalysis < -0.3 and sentimentanalysis >= -0.6):
                sentimenttext = "Negative Book."
            if(sentimentanalysis < -0.1 and sentimentanalysis >= -0.3):
                sentimenttext = "Nuetral with a little bit of Negativity Book."
            if(sentimentanalysis < 0.1 and sentimentanalysis >= -0.1):
                sentimenttext = "Very Nuetral Book."
            if(sentimentanalysis < 0.3 and sentimentanalysis >= 0.1):
                sentimenttext = "Nuetral with a little bit of Positivity Book."
            if(sentimentanalysis < 0.6 and sentimentanalysis >= 0.3):
                sentimenttext = "Positive Book."
            if(sentimentanalysis >= 0.6):
                sentimenttext = "Very Positive Book."

        
        textbox = tk.Text(root, height=30, width=250, padx=15, pady=15) #how big the text box will be
        textbox.insert(1.0, finalsummary) #text of final summary
        textbox.insert(1.0, sentimenttext) #text of sentiment analysis
        textbox.tag_configure("center", justify="center")
        textbox.tag_add("center", 1.0, "end")
        textbox.grid(column=1, row=3)

        browsetext.set("Browse")
        

#browse button
browsetext = tk.StringVar()
browse_btn = tk.Button(root, textvariable=browsetext, command=lambda:openfile(), font="Raleway", bg="#20bebe", fg="white", height=2, width=15) #calling the openfile() function
browsetext.set("Browse")
browse_btn.grid(column=1, row=2)

canvas = tk.Canvas(root, width=1200, height=400)
canvas.grid(columnspan=3)

root.mainloop()
