
import nltk
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN, Birch
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import string
import time
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from sklearn.neighbors import VALID_METRICS, KNeighborsClassifier

# Start the timer
start_time = time.time()

# load data from CSV file into a pandas dataframe
df = pd.read_csv('data/verksamheter.csv')

#####deleteeee
business_descriptions = df['VERKSAMHET'].tolist()

##Tokenize text           bolaget aktiebolaget
sw = set(nltk.corpus.stopwords.words('swedish')).union({'avses','verksamheten','verksamheten','verksamheten','kommer','består','övrig','dock','förvalta','gällande','enlighet','sker','helt','övrigt','ning','lätta','jämte','annat','ge','användas','övriga','samband','dädrmed','området','förenlig','ning','dessutom','via','etc','inkommande','företaget','utom','främst','vidare','områden','där','företag','bolag','såsom','avseende','aktiebolagets','genom','andra','ska', 'idka', 'även', 'utföra', 'annan', 'driva', 'aktiebolaget', 'skulle', 'skulle', 'bolaget', 'bolagets', 'bedriva', 'ävensom', 'samt', 'Bedriva', 'verksamhet', 'skall', 'förenlig', 'därmed', 'erbjuder', 'inom' , 'äga'})#.union({'ska', 'skulle', 'samt'})

#sw = {'ska'}

print('ska' in sw)
 
# preprocess data
def preprocess_text2(text:str):
    # tokenize text into words
    text = str(text)
    words = nltk.word_tokenize(text.lower())
    # remove stop words and punctuation
    words = [word for word in words if word not in sw and word not in string.punctuation and not any(w in "0123456789" for w in word)]
    # join words back into a single string
    return ' '.join(words)

df['t_processed_description'] = df['VERKSAMHET'].apply(preprocess_text2)
print("done1")

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


# transform descriptions into feature vectors using TF-IDF vectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['t_processed_description'])
print("done2")

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X.toarray())


# Normalize the feature vectors to have unit length
X = StandardScaler(with_mean=False).fit_transform(X)

# train DBSCAN model on the feature vectors 0.081  9c     eps=0.9, min_samples=2 ger 22 c
# train DBSCAN model on the feature vectors    euclidean  cosine  eps=0.09, min_samples=2
dbscan = DBSCAN(metric='cosine',eps=0.003, min_samples=22, algorithm='auto' )#,eps=0.08, min_samples=8, algorithm='auto')# auto  'kd_tree', 'brute', 'auto', 'ball_tree'
#dbscan = DBSCAN(metric='cosine',eps=0.13, min_samples=15, algorithm='auto' )#,eps=0.08, min_samples=8, algorithm='auto')# auto  'kd_tree', 'brute', 'auto', 'ball_tree'
labels = dbscan.fit(X)####################
labelss = dbscan.labels_


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
plt.show()

print("labelss",labelss)
ddd = list(labelss)
print("labelss",ddd)
print("max max labelss",max(ddd))
print("max max labelss",max(ddd))
print("max max labelss",max(ddd))

print("len(labelss) ",len(labelss))

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labelss)) - (1 if -1 in labelss else 0)
n_noise_ = list(labelss).count(-1)
print("n_clusters_",n_clusters_)
print("n_noise_" , n_noise_)
print("done3")

df['labels_1'] = labelss
print(list(df.groupby('labels_1').size()))    #[10, 3, 28, 52, 100, 
print(" ")


core_labels = dbscan.labels_[dbscan.core_sample_indices_]
print(core_labels) # de som fick cluster
dddff = list(core_labels)

print(len(core_labels)) # antal de som fick cluster

# extract business descriptions for core samples
core_sample_indicesss = dbscan.core_sample_indices_
core_samplese = df.iloc[core_sample_indicesss]['VERKSAMHET']


print()
core_samplese = core_samplese.to_list()
#print(core_samplese[0])
#print(core_samplese[1])

df_new = pd.read_csv('data/output_file.csv')#.head(20) # data/output_file.csv om vi ska köra det in jept

clf = KNeighborsClassifier(metric='cosine')
clf.fit(X[dbscan.core_sample_indices_,:], core_labels) #center.values(), center.keys()       X,labels
#new_labels = clf.predict(X)   #predict       new_X.toarray()
vectors = df_new.apply(lambda x: vectorizer.transform([preprocess_text2(x['VERKSAMHET'])]).toarray()[0], axis=1).values.tolist()
#print(vectors)
new_labels = clf.predict(vectors)   #predict       new_X.toarray()
print(clf.predict_proba(vectors))
new_labels_2 = clf.predict_proba(vectors).argsort(axis=1)[:, -2] #[:, -2:]
print("new_labels  ",new_labels.tolist() )
df_new['labels'] = new_labels
print(df_new)

print("new_labels  ",new_labels_2.tolist() )
df_new['labels_2'] = new_labels_2
print(df_new)

print(list(df_new.groupby('labels').size()))    #[10, 3, 28, 52, 100, 
print(" ")
print(list(df_new.groupby('labels_2').size()))    #[10, 3, 28, 52, 100, 
print(" ")

labb=df_new['labels'].values.tolist()
print("done4")
print(labb)
print()
print("ax(dddff)", max(dddff))
#the second cluster
labb22 = df_new['labels_2'].values.tolist()
print(labb22)

print(len(labb))
print(len(labb22))

print()

import numpy as np

new_business_descriptions = df_new['VERKSAMHET'].tolist()
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

    # Vectorize the new business description
    new_description_vec = vectorizer.transform([new_business_description])
    # Predict the cluster of the new business description
    predicted_cluster = labb[iii]
    print("predicted_cluster: " ,predicted_cluster)

    # Get the descriptions in the predicted cluster
    cluster_descriptions = []
    for i in range(len(core_samplese)):
        if dddff[i] == predicted_cluster:
            cluster_descriptions.append(core_samplese[i])
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
  
    for i in range(3):
        print(f"{cluster_descriptions[max_scores_indices[0][i]]}: {similarity_scores[0][max_scores_indices[0][i]]}")
    

    max_scores_indices = np.argsort(-similarity_scores)
    #print("Top 3 similarity scores:")
    #for i in range(3):
    #    print(f"{cluster_descriptions[max_scores_indices[0][i]]}: {similarity_scores[0][max_scores_indices[0][i]]}")
    
    predicted_cluster = labb22[iii]
    print("predicted_cluster: " ,predicted_cluster)
    #iii += 1
    #merged_list = []
    iii += 1

    # The second        The second     The second    The second    The second           
    # Get the descriptions in the predicted cluster
    cluster_descriptions = []
    for i in range(len(core_samplese)):
        if dddff[i] == predicted_cluster:
            cluster_descriptions.append(core_samplese[i])
        #iii += 1
    #print("predicted_cluster: " ,predicted_cluster)
    # Compute the cosine similarity between the new description and the descriptions in the predicted cluster
    similarity_scores = cosine_similarity(new_description_vec, vectorizer.transform(cluster_descriptions))
    # Print the maximum similarity score
    print("Max similarity score:", np.max(similarity_scores))
    maxx = np.max(similarity_scores)
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

    for i in range(3):
        print(f"{cluster_descriptions[max_scores_indices[0][i]]}: {similarity_scores[0][max_scores_indices[0][i]]}")
    
    max_scores_indices = np.argsort(-similarity_scores)
     

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


# Skriv ut antalet förekomster av varje cluster
antal_n = []
tom = 0
for i in range(0,max(labb)):
    count_1 = labb.count(i)
    antal_n.append(count_1)
    if count_1 == 0:
        tom +=1
        #print("class : ", i , "är tomt")
print("antal_prediktion i varje cluster : ", antal_n)
print("len antal_prediktion  : ", len(antal_n))
print("antal toma cluster : ", tom)

# create a list of two lists
combined_list = []
for i in range(len(labb)):
    combined_list.append(labb[i])
    combined_list.append(labb22[i])
    
    
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
print("(Second cluster) Nummber of descriptions that get similarity  equal or over to 30 procent to the cluster that it belong to :" , rr2 , " av ",len(df_new) )
print(" Nummber of descriptions that get similarity equal or over 30 procent to the cluster that it belong to :", len(df_new)-rr2 )
print(" ")
print("(First cluster) Nummber of descriptions that get similarity 0 procent to the cluster that it belong to :" , ww , " av ",len(df_new) )
print("(Second cluster) Nummber of descriptions that get similarity 0 procent to the cluster that it belong to :" , ww2 , " av ",len(df_new) )
print(" ")
print("(First cluster) Nummber of descriptions that has similarity 0 procent within the cluster that it belong to :" , mm , " av ",len(df_new) )
print("(Second cluster) Nummber of descriptions that has similarity 0 procent within the cluster that it belong to :" , mm2 , " av ",len(df_new) )
print(" ")

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

print(" ")
print("lab2 tow c: " , combined_list)

# Skriv ut antalet förekomster av varje cluster
antal_n = []
tom = 0
for i in range(0,max(combined_list)):
    count_1 = combined_list.count(i)
    antal_n.append(count_1)
    if count_1 == 0:
        tom +=1
        #print("class : ", i , "är tomt")
print("combined_list antal_prediktion i varje cluster : ", antal_n)
#print("len antal_prediktion  : ", len(antal_n))
print("antal toma cluster : ", tom)



print(" ")

