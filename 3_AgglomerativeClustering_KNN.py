

import nltk
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score , calinski_harabasz_score 
from sklearn.decomposition import PCA
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import spacy
import string
import matplotlib.pyplot as plt 
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import statistics
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import time
# Start the timer
start_time = time.time()

# Load the data
data = pd.read_csv('data/verksamheter.csv')


#####Start with preprocess text#####
#-1
##lemmatize text
# Load the Swedish language model from spaCy
nlp = spacy.load('sv_core_news_sm')

# Define a function to lemmatize text
def lemmatize_text(text):
    text = str(text)
    doc = nlp(text)
    lemmas = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return ' '.join(lemmas)

#data['lemmatized_description'] = data['VERKSAMHET'].apply(lemmatize_text)

#-2
##Tokenize text
#sw = set(nltk.corpus.stopwords.words('swedish')).union({'Bolaget','Bolagets','avses','verksamheten','verksamheten','verksamheten','kommer','består','övrig','dock','förvalta','gällande','enlighet','sker','helt','övrigt','ning','lätta','jämte','annat','ge','användas','övriga','samband','dädrmed','området','förenlig','ning','dessutom','via','etc','inkommande','företaget','utom','främst','vidare','områden','där','företag','bolag','såsom','avseende','aktiebolagets','genom','andra','ska', 'idka', 'även', 'utföra', 'annan', 'driva', 'aktiebolaget', 'skulle', 'skulle', 'bolaget', 'bolagets', 'bedriva', 'ävensom', 'samt', 'Bedriva', 'verksamhet', 'skall', 'förenlig', 'därmed', 'erbjuder', 'inom' , 'äga'})#.union({'ska', 'skulle', 'samt'})
sw = {'ska'}

print('ska' in sw)
# preprocess data
def preprocess_text(text:str):
    text = str(text)
    # tokenize text into words
    words = nltk.word_tokenize(text.lower())
    # remove stop words and punctuation
    words = [word for word in words if word not in sw and word not in string.punctuation and not any(w in "0123456789" for w in word)]
    # join words back into a single string
    return ' '.join(words)

data['t_processed_description'] = data['VERKSAMHET'].apply(preprocess_text)

# Function to preprocess the text by removing stop words
def preprocess_text123(text):
    text = str(text)
    # Tokenize the text
    words = nltk.word_tokenize(text.lower())
    # Remove stop words
    tokens = [word for word in words if word not in sw and word not in string.punctuation and not any(w in "0123456789" for w in word)]
    # Join the tokens back into a single string
    processed_text = " ".join(tokens)
    return processed_text

#-3   Non-preprocissing
stemmer = PorterStemmer()
def preprocess_text24(text:str):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    #text = [word for word in text.split() if word not in stop_words and not any(w in "0123456789" for w in word)]
    #text = [stemmer.stem(word) for word in text]
    #text = " ".join(text)
    return text

#data['description_processed24'] = data['VERKSAMHET'].apply(preprocess_text24)

business_descriptions = data['t_processed_description'].tolist()
business_descriptions23 = data['t_processed_description']

###########End of preprocess text###########
print("done00")

#print(df['description_processed'])
# Convert the descriptions to feature vectors
vectorizer = TfidfVectorizer()  #TfidfVectorizer(stop_words="english")   # backup
X = vectorizer.fit_transform(data['t_processed_description']) #df["description"]
print("done0")

# Apply PCA to reduce the dimensionality of the data
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X.toarray())
#tsne = TSNE(n_components=2)
#X_tsne = tsne.fit_transform(X.toarray())
print("done1")

'''
scores = []
max_rang = 30  # chose the max range of the nummber of clusters
for n_clusters in range(2, max_rang):    # it take to -1  number of clusters give us 0,60  om det blir 5 så ger det oss 0,80
    hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
    labels = hierarchical.fit_predict(X.toarray())
    score = silhouette_score(X, labels)
    print(n_clusters,"-----silhouette_score-----", score)
    scores.append(score)
optimal_n_clusters = scores.index(max(scores)) + 2
print("nummber of cluster is : " , optimal_n_clusters)
'''
# Cluster the data using clustering (n_clusters=optimal_n_clusters, affinity = 'cosine',linkage='complete')      (n_clusters=None, linkage=linkage_type, affinity=distance_metric, distance_threshold=0.1)
optimal_n_clusters = 21
hierarchical = AgglomerativeClustering(n_clusters= optimal_n_clusters , metric = 'euclidean',linkage='ward')  #  =optimal_n_clusters, affinity = 'euclidean',linkage='ward'  n_clusters=optimal_n_clusters average  None, distance_threshold=0.7
print("done2")
X = X.toarray()
print("done21")
labels = hierarchical.fit_predict(X_pca)
print("done22")

# Plot the clusters
unique_labels = np.unique(labels)
colors = plt.cm.get_cmap('tab20')

for label in unique_labels:
    cluster_points = X_pca[labels == label]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], color=colors(label), label=f'Cluster {label + 1}')

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Agglomerative Clustering Clustering Result with PCA')
plt.legend()

print(labels)

# Predict the cluster labels for new descriptions
new_descriptions = pd.read_csv('output_file.csv')# .head(10)  # data/data_not_over    output_file   ('data/data_not_over.csv')#.head(20)
# Preprocess the new input descriptions  lemmatize_text
#new_descriptions['description_processed'] = new_descriptions['VERKSAMHET'].apply(lemmatize_text)

# Vectorize the new input descriptions
# Preprocess the new input descriptions
new_X = vectorizer.transform(new_descriptions['VERKSAMHET'])

# Apply PCA to reduce the dimensionality of  the new input descriptions
pca = PCA(n_components=2)
Y_pca = pca.fit_transform(new_X.toarray())
print("done3")

clf = KNeighborsClassifier()
clf.fit(X,labels) #center.values(), center.keys()       X,labels
new_labels = clf.predict(new_X)   #predict       new_X.toarray()
print("new_labels  ",new_labels )
print("done4")

new_labels_2 = clf.predict_proba(new_X).argsort(axis=1)[:, -2] #[:, -2:]

print("new_labels  ",new_labels.tolist() )
new_descriptions['labels'] = new_labels
fdfff = new_labels.tolist()
print(new_descriptions)

print("new_labels 2  ",new_labels_2.tolist() )
new_descriptions['labels_2'] = new_labels_2
print(new_descriptions)

fdf = new_labels_2.tolist()

# create a list of two lists
combined_list = []
for i in range(len(fdfff)):
    combined_list.append(fdfff[i])
    combined_list.append(fdf[i])
    
print(" ")
labb=new_descriptions['labels'].values.tolist()
print("done4")
print(labb)
print()
print(list(new_descriptions.groupby('labels').size()))    #[10, 3, 28, 52, 100, 
print(" ")
#the second cluster
labb22 = new_descriptions['labels_2'].values.tolist()
print(labb22)
print(len(labb))
print(len(labb22))
print(list(new_descriptions.groupby('labels_2').size()))    #[10, 3, 28, 52, 100, 
print(" ")
print()


resulta_list=[]
# Print the predicted cluster labels for new descriptions
for i, description in enumerate(new_descriptions['VERKSAMHET']):
    resulta_list.append(new_labels[i])
    #print("="*50)
print("="*50)

print("resulta_list", resulta_list)
#print("resulta_list len", len(resulta_list))





import numpy as np

new_business_descriptions = new_descriptions['VERKSAMHET'].tolist()
rr = 0
rr2 = 0
iii = 0
ww = 0
ww2 = 0
mm = 0
mm2 = 0
lab1 = []
lab2 = []
maxx11 = []
maxx22 = []
min11 = []
min22 = []

# Loop through all new business descriptions
for new_business_description in new_business_descriptions:
    print("---------------------")
    print('Input data:', new_business_description)


    new_business_description = preprocess_text123(new_business_description)   #preprocess_text123
    print('Input data 2:', new_business_description)

    # Vectorize the new business description
    new_description_vec = vectorizer.transform([new_business_description])
    # Predict the cluster of the new business description
    predicted_cluster = labb[iii]
    print("predicted_cluster: " ,predicted_cluster)

    # Get the descriptions in the predicted cluster
    cluster_descriptions = []
    for i in range(len(business_descriptions)):
        if labels[i] == predicted_cluster:
            cluster_descriptions.append(business_descriptions[i])
        #iii += 1
    #print("predicted_cluster: " ,predicted_cluster)
    # Compute the cosine similarity between the new description and the descriptions in the predicted cluster
    similarity_scores = cosine_similarity(new_description_vec, vectorizer.transform(cluster_descriptions))
    # Print the maximum similarity score
    print("Max similarity score:", np.max(similarity_scores))
    maxx = np.max(similarity_scores)
    maxx11.append(maxx)
    #similarity_scores222 = np.append(similarity_scores222, [maxx])
    if maxx >=0.3:
        print("yes, it is similar and in the same cluster")
        rr += 1
    maxx = np.max(similarity_scores)
    if maxx ==0:
        ww += 1
    
    maxx = np.min(similarity_scores)
    if maxx ==0:
        mm += 1

    # Print the similarity scores
    print("Similarity scores:")
    print("")
    # Sort the list in descending order based on similarity scores
    # Create a list of tuples containing the similarity score and corresponding text
    similarity_texts = list(zip(similarity_scores[0], cluster_descriptions))

    # Sort the list in descending order based on similarity scores
    similarity_texts = sorted(similarity_texts, key=lambda x: x[0], reverse=True)

    # Print the top 3 highest similarity scores and their corresponding texts in the cluster
    for i in range(min(3, len(similarity_texts))):
        score, cluster_text = similarity_texts[i]
        print(f"Similarity score: {score}")
        print(f"Corresponding text: {cluster_text}")
        print()

    print("******")
    
    max_scores_indices = np.argsort(-similarity_scores)
    #print("Top 3 similarity scores:")
    #for i in range(3):
    #    print(f"{cluster_descriptions[max_scores_indices[0][i]]}: {similarity_scores[0][max_scores_indices[0][i]]}")
    
    predicted_cluster = labb22[iii]
    print("predicted_cluster: " ,predicted_cluster)
    #iii += 1
    #merged_list = []
    iii += 1

    # The second                
    # Get the descriptions in the predicted cluster
    cluster_descriptions = []
    for i in range(len(business_descriptions)):
        if labels[i] == predicted_cluster:
            cluster_descriptions.append(business_descriptions[i])
        #iii += 1
    #print("predicted_cluster: " ,predicted_cluster)
    # Compute the cosine similarity between the new description and the descriptions in the predicted cluster
    similarity_scores = cosine_similarity(new_description_vec, vectorizer.transform(cluster_descriptions))
    # Print the maximum similarity score
    print("Max similarity score:", np.max(similarity_scores))
    maxx2 = np.max(similarity_scores)
    maxx22.append(maxx2)

    if maxx2 >=0.3:
        print("yes, it is similar and in the same cluster")
        rr2 += 1
    
    maxx2 = np.max(similarity_scores)
    if maxx2 ==0:
        ww2 += 1

    maxx = np.min(similarity_scores)
    if maxx ==0:
        mm2 += 1
    # Print the similarity scores
    print("Similarity scores:")
    print("")

    # Create a list of tuples containing the similarity score and corresponding text
    similarity_texts = list(zip(similarity_scores[0], cluster_descriptions))

    # Sort the list in descending order based on similarity scores
    similarity_texts = sorted(similarity_texts, key=lambda x: x[0], reverse=True)

    # Print the top 3 highest similarity scores and their corresponding texts in the cluster
    for i in range(min(3, len(similarity_texts))):
        score, cluster_text = similarity_texts[i]
        print(f"Similarity score: {score}")
        print(f"Corresponding text: {cluster_text}")
        print()

    print("************************************")
    
    max_scores_indices = np.argsort(-similarity_scores)
    
print(" ")

plt.show()

# Suppose results is your list of similarity scores
results = np.array([maxx11])

# Create bins for the plot
bins = [0, 0.33, 0.67, 1]

# Create labels for the bins
labels1 = ['0-0.33', '0.33-0.67', '0.67-1']

# Use np.histogram to get counts per bin
counts, _ = np.histogram(results, bins)

# Generate the plot
plt.figure(figsize=(10, 5))
plt.bar(labels1, counts, color=['red', 'yellow', 'green'])
plt.xlabel('Similarity Score Range')
plt.ylabel('Count')
plt.title('Distribution of Similarity Scores')

print("Print counts when we look at one cluster")

# Print counts
for i, count in enumerate(counts):
    print(f"Range {bins[i]}-{bins[i+1]}: {count} items")

plt.show()

print()
# Suppose results is your list of similarity scores
results22 = np.array([maxx22])

# Create bins for the plot
bins = [0, 0.33, 0.67, 1]

# Create labels for the bins
labels1 = ['0-0.33', '0.33-0.67', '0.67-1']

# Use np.histogram to get counts per bin
counts, _ = np.histogram(results22, bins)

# Generate the plot
plt.figure(figsize=(10, 5))
plt.bar(labels1, counts, color=['red', 'yellow', 'green'])
plt.xlabel('Similarity Score Range')
plt.ylabel('Count')
plt.title('Distribution of Similarity Scores')


print("Print counts when we look at tow cluster")
# Print counts
for i, count in enumerate(counts):
    print(f"Range {bins[i]}-{bins[i+1]}: {count} items")

plt.show()
print()

print(" ")
print("(Second cluster) Nummber of descriptions that get similarity  equal or over to 30 procent to the cluster that it belong to :" , rr2 , " av ",len(new_descriptions) )
print(" Nummber of descriptions that get similarity equal or over 30 procent to the cluster that it belong to :", len(new_descriptions)-rr2 )
print(" ")
print("(First cluster) Nummber of descriptions that get similarity 0 procent to the cluster that it belong to :" , ww , " av ",len(new_descriptions) )
print("(Second cluster) Nummber of descriptions that get similarity 0 procent to the cluster that it belong to :" , ww2 , " av ",len(new_descriptions) )
print(" ")
print("(First cluster) Nummber of descriptions that has similarity 0 procent within the cluster that it belong to :" , mm , " av ",len(new_descriptions) )
print("(Second cluster) Nummber of descriptions that has similarity 0 procent within the cluster that it belong to :" , mm2 , " av ",len(new_descriptions) )
print(" ")

print(" ")


# create a list of two lists
combined_list = []
for i in range(len(labb)):
    combined_list.append(labb[i])
    combined_list.append(labb22[i])
    
print(" ")
print("lab1 : " , labb)
print("lab2 : " , combined_list)

print(" ")

#print(new_descriptions)
print(" ")

data['labels_0'] = labels

# Count the occurrences of each number
counts1 = Counter(labels)
# Print the counts
labbb=[]
for number, count in counts1.items():
    #print(f"Number {number} appears {count} times")
    labbb.append(count)
print("lab 222 input : " , labbb)

print(" ")

# Print the size of each cluster
print("Print the size of each pri cluster")
print(list(new_descriptions.groupby('labels').size()))    #[10, 3, 28, 52, 100, 
print(" ")
print(list(new_descriptions.groupby('labels_2').size()))    #[10, 3, 28, 52, 100, 
print(" ")

data1 = list(data.groupby('labels_0').size())
mean = statistics.mean(data1)
std_dev = statistics.stdev(data1)
print("Mean input:", mean)
print("Standard Deviation:", std_dev)

data2 = list(new_descriptions.groupby('labels').size())
mean = statistics.mean(data2)
std_dev = statistics.stdev(data2)
print("Mean input new first :", mean)
print("Standard Deviation:", std_dev)

data3 = list(new_descriptions.groupby('labels_2').size())
mean = statistics.mean(data3)
std_dev = statistics.stdev(data3)
print("Mean input new second ::", mean)
print("Standard Deviation:", std_dev)
print(" ")

#exit()
##Result list
#my_list = resulta_list
my_list = labb #[2,2,3,3,5,6,8,8,9,8]  # Example list
index1 = 0                                # Initialize first index
index2 = 1                                # Initialize second index
count = 0                                 # Initialize count of matching values
time = 2                                  # Specify amount of time to move indices

while index2 < len(my_list):              # While second index is within bounds of the list
    
    if my_list[index1] == my_list[index2]:  # If the values at the two indices are the same
        count += 1                           # Increment the count
        index1 += time                       # Move both indices by the specified time
        index2 += time
    else:
        index1 += time                          # Move both indices by 1
        index2 += time

print( "len of my_list ", len(my_list))
wrong_predict = (len(my_list) / 2) - count
print(f"The nummber of correct prediction is: {count}")  # Print the count of matching values
wrong_predict = int(wrong_predict)
print("The nummber of wrong prediction is: " , wrong_predict)

precent = (100/(len(my_list)/2)) * count

print(precent , "%" , "  is correct" )
print(" ")
    
#Tow cluster result list
my_list = combined_list
#my_list = [2,2,3,5,5,6,8,6,9,8,4,5]  # Example list
index1 = 0                              # Initialize first index
index11 = 1                                 
index2 = 2                              # Initialize second index
index3 = 3                             
count = 0                                 # Initialize count of matching values
time = 4                                  # Specify amount of time to move indices

while index2 < len(my_list):              # While second index is within bounds of the list
    
    if (my_list[index1] == my_list[index2]) or (my_list[index1] == my_list[index3]) or (my_list[index11] == my_list[index3]) or (my_list[index11] == my_list[index3]) :  # If the values at the two indices are the same
        count += 1                           # Increment the count
        index1 += time                       # Move both indices by the specified time
        index11 += time
        index2 += time                       # Move both indices by the specified time
        index3 += time
    else:
        index1 += time                       # Move all indices by the specified time
        index11 += time
        index2 += time                       
        index3 += time

print( "len of merged_list ", len(my_list))
wrong_predict = (len(my_list) / 4) - count
print(f"The nummber of correct prediction is: {count}")  # Print the count of matching values
wrong_predict = int(wrong_predict)
print("The nummber of wrong prediction is: " , wrong_predict)
precent = (100/(len(my_list)/4)) * count
print(precent , "%" , "  is correct" )
print(" ")
print(" ")

print(" ")

print("lab 0 : " , labels)

print(" ")
print(" lab 0 antal i varje cluster lab 0 ",list(data.groupby('labels_0').size()))    #[10, 3, 28, 52, 100, 
print(" ")


print(" ")
print("lab1 one c: " , labb)

# Skriv ut antalet förekomster av varje cluster
antal_n = []
tom = 0
for i in range(0,max(labb)):
    count_1 = labb.count(i)
    antal_n.append(count_1)
    if count_1 == 0:
        tom +=1
        #print("class : ", i , "är tomt")
print("labb antal_prediktion i varje cluster : ", antal_n)
#print("len antal_prediktion  : ", len(antal_n))
print("antal toma cluster : ", tom)

print(" ")
print("lab2 tow c: " , combined_list)
print(" ")
antal_n = []
tom = 0
for i in range(0,max(combined_list)):
    count_1 = combined_list.count(i)
    antal_n.append(count_1)
    if count_1 == 0:
        tom +=1
        #print("class : ", i , "är tomt")
print("combined_list antal_prediktion i varje cluster of merge list: ", antal_n)
#print("len antal_prediktion  : ", len(antal_n))
print("antal toma cluster : ", tom)

# Step 3: Analyze the clusters
for i in range(hierarchical.n_clusters):
    cluster_documents = [doc for doc, labels in zip(business_descriptions23, labels) if labels == i]
    words = ' '.join(cluster_documents).split()
    counter = Counter(words)
    most_common_words = counter.most_common(10)  # Top 5 words, adjust as needed
    print(f"Cluster {i}: {', '.join(word for word, freq in most_common_words)}")

import time
# End the timer
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds")


exit()


