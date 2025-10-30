import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
from sklearn.model_selection import train_test_split as tts
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
from wordcloud import WordCloud

# printing dataset
data = pd.read_csv(r"C:\Users\DeLL\OneDrive\Desktop\VIPS\2nd Year\4th Sem\FML\FML MINI PROJECT\FML MINI PROJECT\spam.csv")
print(f'Sample Data in the dataset:\n{data.head(1)}\n')
print(f'Total number of rows = {data.shape[0]}')
print(f'Total number of columns = {data.shape[1]}\n')

# data imputation
data.drop_duplicates(inplace=True)
# when we are performing the operations directly on our dataset
# and not assigning to a new variable, then inplace=True is used
print("After removing duplicates -\n")
print(f'Total number of rows = {data.shape[0]}')
print(f'Total number of columns = {data.shape[1]}\n')
if data.isnull().sum().sum() == 0 : 
  # it gives sum of total null values in dataframe (1st sum(): col'ns; 2nd sum(): in whole data frame)
  print("There are no null values in the dataset.\n")

# redefining ham and spam
data['Category'] = data['Category'].replace(['ham', 'spam'], ['Not Spam', 'Spam'])
print(f'Sample Data in the dataset:\n{data.head()}\n')

msg = data['Message'] # input (independent variable) given
cat = data['Category'] # output (dependent variable) to be predicted

# train_test_split
(X_train, X_test, y_train, y_test) = tts(msg, cat, test_size = 0.2, random_state = 4)
print(f'Training Set (X_train): \n {X_train.head(1)} \n')
print(f'Testing Set (X_test): \n {X_test.head(1)} \n')
print(f'Training Set (y_train): \n {y_train} \n')
print(f'Testing Set (y_test): \n {y_test} \n')

# CountVectorizer converts text data into numerical data
cv = CountVectorizer(stop_words='english') # cv is an object here
# stop_words are like a, an, the, in etc.. We are eliminating these words
#as they doesn't give much importance while classifying emails as spam

# converting input training data to numerical format
X_train_num = cv.fit_transform(X_train)

# feature scaling not required as we are not having any numerical data that needs to be in range
# CountVectorizer is used instead of One-Hot Encoding

# Training the Naive Bayes model on training set (MultinomialNB() due to discrete data - spam (1) or not spam (0))
model = MultinomialNB()
model.fit(X_train_num, y_train)

# printing performance metrics
X_test_transformed = cv.transform(X_test) # transforms your text data into numeric vectors
print(f'Accuracy score is: {model.score(X_test_transformed, y_test)*100} %\n') # model.score takes input features (X) and true labels (y), not predictions (like y_pred)

# Generating classification report
y_pred = model.predict(X_test_transformed)
report = classification_report(y_test, y_pred, target_names=['Not Spam', 'Spam'])
print("Classification Report is - \n",report)

# Confusion Matrix
st.markdown("<h1 style='font-size:28px;'>Heatmap -</h1>", unsafe_allow_html=True)
conf_matrix = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(4,3))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Greens', xticklabels=['Not Spam', 'Spam'], yticklabels=['Not Spam', 'Spam'])
plt.ylabel('True label')
plt.xlabel('Predicted label')
st.pyplot(fig)

# Word Cloud
all_messages = " ".join(data['Message'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_messages)
# A WordCloud is a visual representation of text data where
# Words that appear more frequently in the data are shown in larger font sizes.
# It's a quick and intuitive way to understand the most common or important words in a dataset.

# Displaying the word cloud
st.markdown("<h1 style='font-size:28px;'>Word Cloud -</h1>", unsafe_allow_html=True)
st.image(wordcloud.to_array(), use_container_width=True)

# predicting Spam or Not Spam
def predict (message) :
  input = cv.transform([message]).toarray()
  result = model.predict(input)
  return result

# Streamlit is an open-source Python framework that lets you build interactive web apps for machine learning and data science 
# projects — super quickly and easily, without needing to know HTML, CSS, or JavaScript.
st.header('Spam Email Detection')
input_msg = st.text_input('Enter Message')
if st.button('Check'):
    if input_msg.strip() != "":
        output = predict(input_msg)
        st.success(f"Prediction: {output[0]}")
    else:
        st.warning("Please enter a message.")
