# all necessary imports
import numpy as np 
import pandas as pd 
from IPython.display import display
import re
from nltk.corpus import stopwords
from sklearn.metrics import log_loss
import string
import spacy
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import nltk
import ssl
import pickle
stopwords = stopwords.words('english')

#Loading the dataset consisting of various questions related to football world-championships
data = {'query':['Who became the second place in 1974?', 'Which team became 6th place in 2006?', 'Which team was world champion in the year 2010?', 'Who won in 1990?', 'Who was the winner in 1934?',
        'Who became the world champion in 1938?', 'Which team won in 1974?', 'Which team was on the 5th place in the worldcup in 1950?',
        'Who got the 2nd place in 1986?', 'Who was the world-champion in the world-championship of 2018?', 'Which team is the champion of the year 2002?',

        'On which position was Germany in 1986?', 'Brasil got which place in 2014?', 'Which place was Morroco on in 2022?', 'What was the ranking of Sweden in 1958?', 
        'On which place did the Netherlands land on in the year 1974?', 'What was the placement of Italy in 1938?', 'What ranking did England have in the world-championship of 2010?',
        'What was Spain\'s ranking in 2018?', 'What was the placement of Poland in 2002?', 'Which place did France get in the worldcup 2018?',

        'Who hosted the world cup in the year 2018?', 'Which nation was the host of the world cup in 1950?', 'Where was the world cup 1958?', 'Where was the world cup held in 1958?',
        'Who was the host of the world-championship in 2002?', 'Which country hosted the world cup in the year 1978?', 'Where was the world cup held in 1990?',
        'Where did the world cup take place in the year 2006?', 'Who has hosted the worldcup in 1974?', 'In which country did the world cup take place in 2022?',

        'On average, how many goals were scored in 1998?', 'How many goals were scored in the world-championship of 1962 on average?', 'What were the average goals scored in 1986?', 'What was the average number of goals scored in 1930?',
        'What was the average of goals in 1974?', 'How many goals have been scored on average in the year 2002? ', 'How many goals were scored on average in 1990?',
        'In the worldcup 1962, how many goals have been scored on average?', 'What was the average number of goals scored in the year 1962?', 'What was the average number of goals in 1938?', 'What amount of goals was scored in 1990 in average?',

        'How many goals were scored 1950?', 'How many goals in the world-championship of 2010?', 'What is the number of overall goals in 2022?', 'How many goals were made in 2002?',
        'How many goals were accumulated in 2018?', 'What was the total of goals in 1960?', 'What was the total number of goals in 1930?',
        'What is the sum of all goals scored in the year 1938?', 'What is the total number of goals scored in the worldcup of 1950?', 'How many goals were scored in the world cup in 1990?', 'What amount of goals was scored in 1990?',

        'How many matches were played in the world-championship of 1930?','What is the number of matches played in 1954?','What was the amount of matches in the 2010 worldcup?', 'How many matches were played in Germany in 2006?',
        'What is the total number of games played in the year 2002?', 'What is the sum of matches played in 1974?', 'In 2018, how many matches took place?',
        'How many matches have been played in 1950?', 'How many games were played in 1974?', 'What is the total of matches played in 1978?',

        'Who was the winner of the world cup in 1990?','Name the champion of 2000?','Which nation was the world champion in the year 2014?','Which nation was the world-champion in the year 2014?','Which team was the first place in 2002?', 'Who won in 2010?',
        'Who won the world cup in 1974?', 'Who was world-champion in 1950?', 'Who got the first place of the world cup in 2002?',
        'Who was the winner of the world-championship in 2006?', 'Who was the champion of the worldcup in 2010?',

        'Hello.', 'Hellooo', 'Hi!', 'Hey,', 'Heyyy,', 'Good day!', 'Good evening.',

        'Thank you!', 'Thanks', 'Thank you very much!', 'Okay, thanksie!',

        'Goodbye', 'Bye', 'See you!', 'Byebyeee', 'Have a nice day'],


       'category':['PlacementTeam', 'PlacementTeam','PlacementTeam','PlacementTeam','PlacementTeam',
       'PlacementTeam','PlacementTeam','PlacementTeam',
       'PlacementTeam','PlacementTeam','PlacementTeam',

       'TeamPlacement','TeamPlacement','TeamPlacement','TeamPlacement',
       'TeamPlacement','TeamPlacement','TeamPlacement',
       'TeamPlacement','TeamPlacement','TeamPlacement',

       'YearHost','YearHost','YearHost','YearHost',
       'YearHost','YearHost','YearHost',
       'YearHost','YearHost','YearHost',

       'year(avg)Goals','year(avg)Goals','year(avg)Goals','year(avg)Goals',
       'year(avg)Goals','year(avg)Goals','year(avg)Goals',
       'year(avg)Goals','year(avg)Goals','year(avg)Goals','year(avg)Goals',

       'yearGoals','yearGoals','yearGoals','yearGoals',
       'yearGoals','yearGoals','yearGoals',
       'yearGoals','yearGoals','yearGoals','yearGoals',

       'yearMatches','yearMatches','yearMatches','yearMatches',
       'yearMatches','yearMatches','yearMatches', 
       'yearMatches','yearMatches','yearMatches', 

       'firstPlace', 'firstPlace', 'firstPlace','firstPlace','firstPlace','firstPlace', 
       'firstPlace','firstPlace','firstPlace',
       'firstPlace', 'firstPlace',
       
       'greeting', 'greeting', 'greeting', 'greeting', 'greeting', 'greeting','greeting',
       
       'thankYou', 'thankYou', 'thankYou', 'thankYou', 
       
       'bye', 'bye', 'bye', 'bye', 'bye']}

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

def train_model():
    df = pd.DataFrame(data)
    
    #necessary nltk and spacy package for the upcoming cleanup part
    nltk.download('stopwords')
    nlp = spacy.load('en_core_web_sm')
    punctuations = string.punctuation

    #clean text using spacy (lemma, lowercasing, removing stopwords)
    def cleanup_text(docs, logging=False):
        texts = [] 
        counter = 1
        for doc in docs:
            if counter % 1000 == 0 and logging:
                print(" %d out of %d documents." % (counter, len(docs)))
            counter += 1
            doc = nlp(doc, disable=['parser', 'ner'])
            tokens = [tok.lemma_.lower().strip() for tok in doc] #Lemmatization and lowercasing
            tokens = [tok for tok in tokens if tok not in stopwords and tok not in punctuations] #removing stopwords
            tokens = ' '.join(tokens)
            texts.append(tokens)
        return pd.Series(texts)
    train_cleaned = cleanup_text(df['query'], logging=True)

        # further cleaning the text to apply a word2vec model
    def cleanup_text_word2vec(docs, logging=False):
        sentences = []
        counter = 1
        for doc in docs:
            if counter % 1000 == 0 and logging:
                print("Processed %d out of %d documents" % (counter, len(docs)))
            doc = nlp(doc, disable=['tagger'])
            doc = " ".join([tok.lemma_.lower() for tok in doc])
            doc = re.split("[\.?!;] ", doc) #splitting text into sentences and words
            doc = [re.sub("[\.,;:!?]", "", sent) for sent in doc] #substitute punctuations 
            doc = [sent.split() for sent in doc]
            sentences += doc
            counter += 1
        return sentences #list of lists of words for the w2v
    train_cleaned_word2vec = cleanup_text_word2vec(df['query'], logging=True)

    #creating a corpus by joining list of lists of words and fit to tf-idf vectorizer (convert to numerical representation)
    corpus = [" ".join(i) for i in train_cleaned_word2vec]
    vectorizer = TfidfVectorizer()
    vec_fit = vectorizer.fit(corpus)



    # Define function to create word vector representation of a given cleaned piece of text by averaging the tf-idf vectors
    # take in cleaned piece of text
    def create_average_vec(doc): 
        #initialize empty average vector of same length as number of features in vec_fit
        average = np.zeros(len(vectorizer.get_feature_names()), dtype='float32')
        num_words = 0.
        #iterate over each word in doc and add tf-idf vector to average vector
        for word in doc.split():
            average = np.add(average, vec_fit.transform([word]).toarray()) 
            num_words += 1.
        if num_words != 0.:
            #divide average vector by number of words to get average vector repr. of text
            average = np.divide(average, num_words)
        return average #return converted cleaned data to use as input for SVC Model
    
    # Create word vectors for entire cleaned dataset
    train_cleaned_vec = np.zeros((df.shape[0], len(vectorizer.get_feature_names())), dtype="float32") 
    for i in range(len(train_cleaned)):
        train_cleaned_vec[i] = create_average_vec(train_cleaned[i])
    
    #encode target variable "category" into numerical values to be used as input for SVC Model.
    Encoder = LabelEncoder()
    y_train = Encoder.fit_transform(df['category'])

    #k-fold cross validation to evaluate the performance of the trained model 
    param_grid = {'C': [0.1, 1, 10, 100, 1000], 
               'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf']} 
    svclassifier = SVC(probability = True)
    #svclassifier.fit(train_cleaned_vec, y_train)
    kfold = KFold(n_splits=5, shuffle=True)
    grid_search = GridSearchCV(svclassifier, param_grid, cv=kfold, refit=True, verbose=3)
    
    #perform grid search by training model with different combinations of specified parameters and evaluate performance
    grid_search.fit(train_cleaned_vec, y_train)

    # Train the model with the best hyperparameters from gridsearch using the cleaned and vectorized data
    svclassifier = SVC(C=grid_search.best_params_['C'],
                   gamma=grid_search.best_params_['gamma'],
                   kernel=grid_search.best_params_['kernel'])
    svclassifier.fit(train_cleaned_vec, y_train)

    # save the model to disk
    filename_model = './picklefiles/intent_detection_model.sav'
    pickle.dump(svclassifier, open(filename_model, 'wb'))

    # save the model to disk
    filename_vecfit = './picklefiles/vec_fit.sav'
    pickle.dump(vec_fit, open(filename_vecfit, 'wb'))

    return 
    
# Code auf Basis von: https://www.kaggle.com/code/taranjeet03/intent-detection-svc-using-word2vec/notebook#)