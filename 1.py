# -*- coding: utf-8 -*-


# Import all packages
import random
import sklearn
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn import metrics
from sklearn.feature_selection import SelectKBest, chi2
from sklearn import preprocessing
from pandas import concat
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from nltk.corpus import stopwords

from nltk import pos_tag

#print('The nltk version is {}.'.format(nltk.__version__))

# Import the file which we will use for classification

# Set folder path to the directory where the files are located
#folder_path = 'C:\\Users\\fujitsu\\Documents\\Python Scripts'

# Start Processing for one article

# Open the JSON data file
#data =  open(os.path.join(folder_path, "MsiaAccidentCases.xlsx"), "r")

data=pd.read_csv("MsiaAccidentCases.csv")
data=pd.read_csv("osha.csv")

#for c in data:
    #text_nopunc+=c.translate(string.maketrans("",""), string.punctuation)+"\"
#print text_nopunc

#print train["Summary Case"]

# Create the list which you would like to filter on
#cause=['Caught in/between Objects', 'Collapse of object', 'Drowning', 'Electrocution', 'Exposure to Chemical Substances','Falls','Fires and Explosion','Other','Struck By Moving Objects','Suffocation']

#caseSummary=data["Summary Case"]
caseTitle=data["Title Case"]

stopwords = stopwords.words('english')


#labelled=[]
#for row in caseSummary:
    #text_nopunc=row.translate(string.maketrans("",""), string.punctuation)
    #text_nopunc=text_nopunc.lower()
    #labelled.append(text_nopunc)
    #labelled = labelled.append(word_tokenize(labelled))
    
    
###############################################################################


#vectorizer2 = TfidfVectorizer(analyzer=u'word',min_df=1)
#x2 = vectorizer2.fit_transform(labelled)
#vectorizer1 = TfidfVectorizer(analyzer=u'word',min_df=1)
#pos = pos_tag(word_tokenize(vectorizer1)) 
#pos = pos_tag(word_tokenize(caseSummary))


#p=[]
#for row in caseSummary:
    #pos = pos_tag(word_tokenize(row))
    #p.append(pos)





##########2222222222222222222222222222222222222222222222222222222222###########
###############################################################################


labelled=[]
for row in caseTitle:
    text_nopunc=row.translate(string.maketrans("",""), string.punctuation)
    #text_nopunc=text_nopunc.lower()
    labelled.append(text_nopunc)
    #labelled = labelled.append(word_tokenize(labelled))
    

p=[]
for row in caseTitle:
    row=row.lower()
    pos = pos_tag(word_tokenize(row))
    p.append(pos)

grammar = r"""
        
        #PP:
            #{<NN|NNP|VB|VBD|VBG|VBN|VBP|VBZ|JJ>*<IN>}
        PP:
            {<NN|NNP|VB|VBD|VBG|VBN|VBP|VBZ|JJ|NNS>*<IN|TO>}
        NP: 
            {<DT|TO>?<NN.*|VB.*|VBN.*|VBG.*|JJ.*|VBD.*|NNP.*>+}
            
            
            
            
            
            
            
            
            
            
        
        
        
        #OBJ:
        # Preposition, Nouns, terminated with Nouns
        #{<IN>*<NN.*>}
        #{<NN><NN>}  
   
   
   
   
        #PF: {<NN|VB|VBD|VBG|VBN|VBP|VBZ>}
        #PP: 
            #{<PF>}
            #{<IN>|<IN><DT>}               # Chunk prepositions followed by NP
        #NP: {<DT|JJ|NN.*>+}          # Chunk sequences of DT, JJ, NN
        
        
        
        
        #NP: 
            #{<PP>}
            #{<NN.*|NNS.*|NNP.*|NNPS.*>+}  
            #{<DT.*|JJ.*|NN.*>+} 
   
        #PF: {<NN|VB|VBD|VBG|VBN|VBP|VBZ>}
        #PP: {<IN>|The.|the.|A.|a.|An.|an.}               # Chunk prepositions followed by NP
        #NP: {<DT|JJ|NN.*>+}          # Chunk sequences of DT, JJ, NN
        #NP: {<NN|NNS|NNP|NNPS>}  
        
        #NBAR:
        #{<NN.*|JJ>*<NN.*>} # Nouns and Adjectives, terminated with Nouns
        #NP:
        #{<NBAR>}
        #{<NBAR><IN><NBAR>} # Above, connected with in/of/etc...
        

        
"""


def leaves(tree):
    """Finds NP (nounphrase) leaf nodes of a chunk tree."""
    for subtree in tree.subtrees(filter = lambda t: t.node=='NP'):
        yield subtree.leaves()
        
def normalise(word):
    """Normalises words to lowercase and stems and lemmatizes it."""
    word = word.lower()
    #word = stemmer.stem_word(word)
    #word = lemmatizer.lemmatize(word)
    return word
    
def acceptable_word(word):
    """Checks conditions for acceptable word: length, stopword."""
    accepted = bool(2 <= len(word) <= 40
        and word.lower() not in stopwords)
    return accepted

def get_terms(tree):
    for leaf in leaves(tree):
        term = [ normalise(w) for w,t in leaf if acceptable_word(w) ]
        yield term


chunker = nltk.RegexpParser(grammar)


n = 0
death_causes = {}
death_words = {}
for r in p:
    #print nltk.ne_chunk(r)
    #print chunker.parse(r)
    tree = chunker.parse(r)
    terms = get_terms(tree)
    t = ""
    for term in terms:
        t = " ".join(term)
        for word in term:
            if word not in death_words:
                death_words[word] = 1
            else:
                death_words[word] += 1
            print word,
        print ","
        if t not in death_causes:
            death_causes[t] = 1
        else:
            death_causes[t] += 1
  






###############################################################################
###############################################################################




vectorizer1 = TfidfVectorizer(analyzer=u'word',min_df=1,  stop_words='english')
x1 = vectorizer1.fit_transform(labelled)
y1 = data['Cause ']


vectorizer = TfidfVectorizer(analyzer=u'word',min_df=1,  stop_words='english')
x = vectorizer.fit_transform(caseSummary)
y = data['Cause ']

#vectorizer.get_feature_names()

    
    
X_train, X_test, y_train, y_test = train_test_split(x1, y1, test_size=0.2, random_state=42)


# Number of features in the Training Set 
X_train.shape

# Number of features in the Test Set
X_test.shape

# Number of documents in Training set
len(y_train)


# Number of documents in Training set
len(y_test)    


# Create the list which you would like to filter on
#cause=['Caught in/between Objects', 'Collapse of object', 'Drowning', 'Electrocution', 'Exposure to Chemical Substances','Falls','Fires and Explosion','Other','Struck By Moving Objects','Suffocation']

    

def train_svm(X, y):
    """
    Create and train the Support Vector Machine.
    """
    svm = SVC(C=10000000.0, gamma=0.5, kernel='rbf')
    svm.fit(X, y)
    return svm



svm = train_svm(X_train, y_train)





# Predict on the Test set
pred = svm.predict(X_test)

# Print the classification rate
print(svm.score(X_test, y_test))

labels = list(set(y_train))






def train_dtc(X, y):
    """
    Create and train the Decision Tree Classifier.
    """
    dtc = DecisionTreeClassifier()
    dtc.fit(X, y)
    return dtc

# Decision Tree Classifier
dt = train_dtc(X_train, y_train)
predDT = dt.predict(X_test)

# Print the classification rate
print(dt.score(X_test, y_test))

csl=caseSummary.tolist()





















clean_review = review_to_words(csl)
print clean_review


def review_to_words(raw_review):
    # Function to convert a raw review to a string of words
    # The input is a single string (a raw movie review), and 
    # the output is a single string (a preprocessed movie review)
    #
    # 1. Remove HTML
    review_text = raw_review
    #
    # 2. Remove non-letters        
    letters_only = re.sub("[^a-zA-Z]", " ", review_text) 
    #
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()                             
    #
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))                  
    # 
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]   
    #
    # 6. Join the words back into one string separated by space, 
    # and return the result.
    return( " ".join( meaningful_words ))















# Remove all the encoding, escape and special characters
#ntext=unicodedata.normalize('NFKD', train["Summary Case"]).encode('ascii','ignore')
#print ntext

# Remove all the punctuations from the text
# def create_tfidf_training_data(docs):
for c in caseSummary:
    text_nopunc=c.translate(string.maketrans("",""), string.punctuation)+" "+text_nopunc
print text_nopunc

# Convert all characters to Lower case
text_lower=text_nopunc.lower()
print text_lower

# Create a stopword list from the standard list of stopwords available in nltk
stop = stopwords.words('english')
print stop

# Remove all these stopwords from the text
text_nostop=" ".join(filter(lambda word: word not in stop, text_lower.split()))
print text_nostop

# Convert the stopword free text into tokens to enable further processing
tokens = word_tokenize(text_nostop)
print tokens



# Create a list with "Terror" Documents and their abstracts
train1=[]
for row in caseSummary.iterrows():
    index, data = row
    train1.append(data.tolist())







