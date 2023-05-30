
# Import necessary libraries
from collections import Counter
import nltk
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
import spacy
import string
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import time

# Start the timer
start_time = time.time()

# Load the business description text data
data = pd.read_csv('data/verksamheter.csv') 

##lemmatize text
# Load the Swedish language model from spaCy
nlp = spacy.load('sv_core_news_sm')

# Define a function to lemmatize text
def lemmatize_text(text):
    text = str(text)
    doc = nlp(text)
    lemmas = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return ' '.join(lemmas)

#-2
##Tokenize text
sw = set(nltk.corpus.stopwords.words('swedish')).union({'gällande','enlighet','sker','helt','övrigt','ning','lätta','jämte','annat','ge','användas','övriga','samband','dädrmed','området','förenlig','ning','dessutom','via','etc','inkommande','företaget','utom','främst','vidare','områden','där','företag','bolag','såsom','avseende','aktiebolagets','genom','andra','ska', 'idka', 'även', 'utföra', 'annan', 'driva', 'aktiebolaget', 'skulle', 'skulle', 'bolaget', 'bolagets', 'bedriva', 'ävensom', 'samt', 'Bedriva', 'verksamhet', 'skall', 'förenlig', 'därmed', 'erbjuder', 'inom' , 'äga'})#.union({'ska', 'skulle', 'samt'})
#sw ={'ska'}
print('ska' in sw)
print('skall' in sw)

print(nlp)
# preprocess data
def preprocess_text(text):
    # tokenize text into words
    words = nltk.word_tokenize(str(text))
    # remove stop words and punctuation
    words = [word for word in words if word not in sw and word not in string.punctuation and not any(w in "0123456789" for w in word)]
    # join words back into a single string
    return ' '.join(words)

data['t_processed_description'] = data['VERKSAMHET'].apply(preprocess_text)   #2000  lab20  20 

# Vectorize the text data using TF-IDF
# replace NaN values with "missing"
data['t_processed_description'].fillna('missing', inplace=True)

business_descriptions = data['t_processed_description'].tolist()

# Vectorize the text data
vectorizer = TfidfVectorizer()   # 
X = vectorizer.fit_transform(business_descriptions)

# Apply PCA to reduce the dimensionality of the data
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X.toarray())
'''
scores = []
max_rang = 28  # chose the max range of the nummber of clusters
for n_clusters in range(20, max_rang):    # it take to -1  number of clusters give us 0,60  om det blir 5 så ger det oss 0,80
    kmeans = KMeans(n_clusters=n_clusters ,  max_iter = 500, n_init= 20)
    labels = kmeans.fit_predict(X)
    score = silhouette_score(X, labels)
    print(n_clusters,"-----silhouette_score-----", score)
    scores.append(score)
optimal_n_clusters = scores.index(max(scores)) + 2
print("nummber of cluster is : " , optimal_n_clusters)

# plot the silhouette scores
plt.plot(range(20, max_rang), scores)
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.show()
'''

optimal_n_clusters = 21
print("nummber of cluster is : " , optimal_n_clusters)
#score = silhouette_score(X, labels)
#print("The silhouette score is : " , score)

# Apply K-means clustering algorithm
kmeans = KMeans(n_clusters=optimal_n_clusters,  max_iter = 500, n_init= 20) # ,n_init='auto'
kmeans.fit(X)

labels = kmeans.fit_predict(X)

# Get the top terms per cluster
order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names_out()


# Plot the clusters
unique_labels = np.unique(labels)
colors = plt.cm.get_cmap('tab20')

for label in unique_labels:
    cluster_points = X_pca[labels == label]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], color=colors(label), label=f'Cluster {label + 1}')

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Agglomerative Clustering Result with PCA')
plt.legend()
plt.show()


# Get the cluster labels for the original data
cluster_labels = kmeans.labels_
# Get the cluster centers (representative descriptions)
cluster_centers = kmeans.cluster_centers_
print("done2")

# Plot the data points and their assigned clusters for the original data
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap="rainbow")
plt.title('Clustered Business Descriptions')
plt.xlabel('PCA Dimension 1')
plt.ylabel('PCA Dimension 2')
plt.show()

####Start Evaluate#####
# Calculate silhouette score   ##The silhouette score measures how similar an object is to its own cluster compared to other clusters. A higher silhouette score indicates better clustering performance. The silhouette score ranges from -1 to 1, with a score of 1 indicating that the object is very similar to its own cluster and very dissimilar to objects in other clusters, and a score of -1 indicating the opposite. We can calculate the average silhouette score for all objects in the dataset to evaluate the overall performance of the clustering model.
silhouette_avg = silhouette_score(X, kmeans.labels_)
print("Silhouette score:", silhouette_avg)

# Calculate Calinski-Harabasz index  ##The Calinski-Harabasz index measures the ratio of the between-cluster variance to the within-cluster variance. A higher value of this index indicates better clustering performance.
ch_score = calinski_harabasz_score(X.toarray(), kmeans.labels_)
print("Calinski-Harabasz index:", ch_score)

# Assign labels to the clusters
data['cluster_labels'] = kmeans.labels_

new_business_df = pd.read_csv('output_file.csv')
# Preprocess the new input descriptions
#new_descriptions['lemmatized_description'] = new_descriptions['VERKSAMHET'].apply(lemmatize_text)    #13
new_business_df['description_processed'] = new_business_df['VERKSAMHET'].apply(preprocess_text)       #11
#new_descriptions['description_processed2'] = new_descriptions['VERKSAMHET'].apply(preprocess_text2)   #13

# Vectorize the new input descriptions
X_new = vectorizer.transform(new_business_df['description_processed'])
# Apply PCA to reduce the dimensionality of the new input descriptions
new_X_pca = pca.transform(X_new.toarray()) ##########
predicted_labels = kmeans.predict(X_new)

print("labels:", predicted_labels)

#with preproc
new_business_descriptions_proc = new_business_df['description_processed'].tolist()

#no preproc
new_business_descriptions = new_business_df['VERKSAMHET'].tolist()

##Result list
#one cluster resulta_list
my_list = predicted_labels #[2,2,3,3,5,6,8,8,9,8]  # Example list
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

print( "len of resulta_list ", len(my_list))
wrong_predict = (len(my_list) / 2) - count
print(f"The nummber of correct prediction is: {count}")  # Print the count of matching values
wrong_predict = int(wrong_predict)
print("The nummber of wrong prediction is: " , wrong_predict)
precent = (100/(len(my_list)/2)) * count
print(precent , "%" , "  is correct" )

print(" ")
import time

time.sleep(3)

rr = 0
rr2 = 0
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
    # Vectorize the new business description
    new_description_vec = vectorizer.transform([new_business_description])

    # Predict the cluster of the new business description
    predicted_cluster = kmeans.predict(new_description_vec)[0]
    lab1.append(predicted_cluster)
    lab2.append(predicted_cluster)

    distances = []
    for centroid in cluster_centers:
        distance = np.linalg.norm(new_description_vec[0] - centroid)
        distances.append(distance)

    closest_cluster = np.argmin(distances)
    a = np.delete(distances, closest_cluster)
    closest_cluster2 = np.argmin(a)
    lab2.append(closest_cluster2)


    # Get the descriptions in the predicted cluster
    cluster_descriptions = []
    for i in range(len(business_descriptions)):
        if cluster_labels[i] == predicted_cluster:
            cluster_descriptions.append(business_descriptions[i])
    print("predicted_cluster: " ,predicted_cluster)
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
   
    max_scores_indices = np.argsort(-similarity_scores)
    #print("Top 3 similarity scores:")
    
    for i in range(3):
        print(f"{cluster_descriptions[max_scores_indices[0][i]]}: {similarity_scores[0][max_scores_indices[0][i]]}")
    
    ########## the second   
    print("the second")
    print("")
    # Get the descriptions in the predicted cluster
    cluster_descriptions = []
    for i in range(len(business_descriptions)):
        if cluster_labels[i] == closest_cluster2:
            cluster_descriptions.append(business_descriptions[i])
    print("predicted_cluster: " ,closest_cluster2)
    # Compute the cosine similarity between the new description and the descriptions in the predicted cluster
    similarity_scores2 = cosine_similarity(new_description_vec, vectorizer.transform(cluster_descriptions))
    # Print the maximum similarity score
    print("Max similarity score:", np.max(similarity_scores2))
    maxx2 = np.max(similarity_scores2)
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
    # Print the maximum similarity score
    #print("Max similarity score:", np.max(similarity_scores))
    # Print the similarity scores
    print("Similarity scores:")
    print("")
    
    for i in range(3): #len(cluster_descriptions)
        print(f"{cluster_descriptions[i]}: {similarity_scores[0][i]}")
    print("******")
    
    max_scores_indices = np.argsort(-similarity_scores2)
    #print("Top 3 similarity scores:")
    
    print("Predicted closest_cluster:", closest_cluster)
    print("Predicted closest_cluster2:", closest_cluster2)
    #print("Second closest cluster:", second_closest_cluster)
print("done5")

print()


# Suppose results is your list of similarity scores
results = np.array([maxx11])

# Create bins for the plot
bins = [0, 0.33, 0.67, 1]

# Create labels for the bins
labels = ['0-0.33', '0.33-0.67', '0.67-1']

# Use np.histogram to get counts per bin
counts, _ = np.histogram(results, bins)

# Generate the plot
plt.figure(figsize=(10, 5))
plt.bar(labels, counts, color=['red', 'yellow', 'green'])
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
labels = ['0-0.33', '0.33-0.67', '0.67-1']

# Use np.histogram to get counts per bin
counts, _ = np.histogram(results22, bins)

# Generate the plot
plt.figure(figsize=(10, 5))
plt.bar(labels, counts, color=['red', 'yellow', 'green'])
plt.xlabel('Similarity Score Range')
plt.ylabel('Count')
plt.title('Distribution of Similarity Scores')


print("Print counts when we look at tow cluster")
# Print counts
for i, count in enumerate(counts):
    print(f"Range {bins[i]}-{bins[i+1]}: {count} items")

plt.show()
print()


#
print(" ")
# Add cluster labels to the dataframe
data['cluster'] = kmeans.labels_

print("Print the size of each cluster")
# Print the size of each cluster
print("Print the size of each cluster" ,list(data.groupby('cluster').size()))    #[10, 3, 28, 52, 100, 
print(" ")

print("One cluster : " , lab1)
print(" ")
new_business_df['cluster1'] = lab1
print(" ")
# Print the size of each cluster
print("Print the size of each cluster" ,list(new_business_df.groupby('cluster1').size()))    #[10, 3, 28, 52, 100, 
print(" ")
print(" ")
print("Tow cluster : " , lab2)
print(" ")
# Print the size of each cluster
# Count the occurrences of each number
counts1 = Counter(lab1)
# Print the counts
labbb=[]
for number, count in counts1.items():
    #print(f"Number {number} appears {count} times")
    labbb.append(count)

print("Print the size of each cluster" ,labbb)

print(" ")
print(" ")



##Result list
#one cluster resulta_list
my_list = lab1 #[2,2,3,3,5,6,8,8,9,8]  # Example list
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

print( "len of resulta_list ", len(my_list))
wrong_predict = (len(my_list) / 2) - count
print(f"The nummber of correct prediction is: {count}")  # Print the count of matching values
wrong_predict = int(wrong_predict)
print("The nummber of wrong prediction is: " , wrong_predict)
precent = (100/(len(my_list)/2)) * count
print(precent , "%" , "  is correct" )

print(" ")
#Tow cluster result list
my_list = lab2
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
import time
# End the timer
end_time = time.time()
# Calculate the elapsed time
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds")
print(" ")


print(" ")

print(" ")
print("(Second cluster) Nummber of descriptions that get similarity  equal or over to 30 procent to the cluster that it belong to :" , rr2 , " av ",len(new_business_df) )
print(" Nummber of descriptions that get similarity equal or over 30 procent to the cluster that it belong to :", len(new_business_df)-rr2 )
print(" ")
print("(First cluster) Nummber of descriptions that get similarity 0 procent to the cluster that it belong to :" , ww , " av ",len(new_business_df) )
print("(Second cluster) Nummber of descriptions that get similarity 0 procent to the cluster that it belong to :" , ww2 , " av ",len(new_business_df) )
print(" ")
print("(First cluster) Nummber of descriptions that has similarity 0 procent within the cluster that it belong to :" , mm , " av ",len(new_business_df) )
print("(Second cluster) Nummber of descriptions that has similarity 0 procent within the cluster that it belong to :" , mm2 , " av ",len(new_business_df) )
print(" ")
print("Silhouette score:", silhouette_avg)
print("Calinski-Harabasz index:", ch_score)

for i in range(optimal_n_clusters):
    print("Cluster %d:" % i)
    worrrd= []
    for ind in order_centroids[i, :10]:  # change 10 to any number of top words you want
        worrrd.append(terms[ind])
    print(' %s' % worrrd)


exit()





