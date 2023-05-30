
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import nltk
import string
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import time
import spacy
from gensim import corpora, models
from sklearn.preprocessing import MinMaxScaler
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.decomposition import LatentDirichletAllocation

# Start the timer
start_time = time.time()

sw = set(nltk.corpus.stopwords.words('swedish')).union({'ning','lätta','jämte','annat','ge','användas','övriga','samband','dädrmed','området','förenlig','ning','dessutom','via','etc','inkommande','företaget','utom','främst','vidare','områden','där','företag','bolag','såsom','avseende','aktiebolagets','genom','andra','ska', 'idka', 'även', 'utföra', 'annan', 'driva', 'aktiebolaget', 'skulle', 'skulle', 'bolaget', 'bolagets', 'bedriva', 'ävensom', 'samt', 'Bedriva', 'verksamhet', 'skall', 'förenlig', 'därmed', 'erbjuder', 'inom' , 'äga'})#.union({'ska', 'skulle', 'samt'})

# Load the business descriptions from a CSV file
business_df = pd.read_csv('data/verksamheter.csv') 
# replace NaN values with "missing"
business_df['VERKSAMHET'].fillna('missing', inplace=True)

data = business_df['VERKSAMHET'].tolist()

##Tokenize text
print('ska' in sw)
# preprocess data
def preprocess_text2222(text:str):
    # tokenize text into words
    words =str(text)
    words = nltk.word_tokenize(words.lower())
    # remove stop words and punctuation
    #words = [word for word in words if word not in sw and word not in string.punctuation and not any(w in "0123456789" for w in word)]
    # join words back into a single string
    return words #' '.join(words)

##Tokenize text
print('ska' in sw)
# preprocess data
def preprocess_text(text:str):
    # tokenize text into words
    text =str(text)
    words = nltk.word_tokenize(text.lower())
    # remove stop words and punctuation
    words = [word for word in words if word not in sw and word not in string.punctuation and not any(w in "0123456789" for w in word)]
    # join words back into a single string
    return words #' '.join(words)

processed_data1 = [preprocess_text(description) for description in data]

#business_df['t_processed_description'] = business_df['VERKSAMHET'].apply(preprocess_text)   #2000  lab20  20 

# Load the Swedish language model from spaCy
nlp = spacy.load('sv_core_news_sm')

# Define a function to lemmatize text
def lemmatize_text(text):
    text = str(text)
    doc = nlp(text)
    lemmas = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return ' '.join(lemmas)

#den ska ändras här och i predict_cluster2
pre_pros = processed_data1

# Create a dictionary and a bag-of-words representation of the data
dictionary = corpora.Dictionary(pre_pros)
corpus = [dictionary.doc2bow(text) for text in pre_pros]

optimal_num_topics = 60 #num_topics_range[np.argmax(coherence_values)]
print(f'Optimal number of topics: {optimal_num_topics}')


# Apply LDA for topic modeling         n_components=num_topics
num_topics = optimal_num_topics  
lda = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=20)

# Transform the corpus to LDA topic distribution space
lda_corpus = lda[corpus]

# Convert lda_corpus to a 2-dimensional array
lda_features = np.zeros((len(lda_corpus), num_topics))

for i, lda_feature in enumerate(lda_corpus):
    for topic_id, topic_prob in lda_feature:
        lda_features[i, topic_id] = topic_prob

# Normalize the LDA features
scaler = MinMaxScaler()
normalized_lda_features = scaler.fit_transform(lda_features)

# Cluster the business descriptions using KMeans
num_clusters = 15 # Specify the number of clusters
kmeans = KMeans(n_clusters=num_clusters, random_state=42 ).fit(normalized_lda_features) # ,  max_iter = 200, n_init= 'auto'  

# Get the cluster labels
cluster_labels = kmeans.labels_

# Get the cluster centers (representative descriptions)
cluster_centers = kmeans.cluster_centers_

labels = kmeans.fit_predict(normalized_lda_features)
print()
print(labels)
#evaluate
# calculate the silhouette score and Calinski-Harabasz index for the clustering
silhouette = silhouette_score(normalized_lda_features, labels)
# calinski_harabasz = calinski_harabasz_score(normalized_lda_features.toarray(), labels)
# print the results
print("Silhouette score:", silhouette)
# print("Calinski-Harabasz index:", calinski_harabasz)
print()


# Apply PCA to reduce the dimensionality of the data
pca = PCA(n_components=2)
X_pca = pca.fit_transform(normalized_lda_features) #.toarray()


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA

# Plot the clusters
unique_labels = np.unique(labels)
colors = plt.cm.get_cmap('tab20')

for label in unique_labels:
    cluster_points = X_pca[labels == label]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], color=colors(label), label=f'Cluster {label + 1}')

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('K-means Clustering Result with PCA')
plt.legend()
plt.show()

# Load the new business descriptions from a CSV file
new_business_df = pd.read_csv('output_file.csv')#.head(10)
new_data = new_business_df['VERKSAMHET'].tolist()


#new_business_df['description_processed'] = new_business_df['VERKSAMHET'].apply(preprocess_text)       #11
#new_business_df['t_processed_description_new'] = new_business_df['VERKSAMHET'].apply(preprocess_text)   #2000  lab20  20 

new_business_descriptions = new_business_df['VERKSAMHET'].tolist()

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
for new_business_description in new_data:
    print("---------------------")
    print('Input data:', new_business_description)
    
    processed_new_description = preprocess_text2222(new_business_description)
    bow_new_description = dictionary.doc2bow(processed_new_description)
    lda_new_description = lda[bow_new_description]
    lda_features_new_description = np.zeros((1, num_topics))

    for topic_id, topic_prob in lda_new_description:
        lda_features_new_description[0, topic_id] = topic_prob

    normalized_lda_features_new_description = scaler.transform(lda_features_new_description)
    # Predict the cluster of the new business description
    predicted_cluster = kmeans.predict(normalized_lda_features_new_description)[0]
    lab1.append(predicted_cluster)
    lab2.append(predicted_cluster)

    #merged_list = []

    distances = []
    for centroid in cluster_centers:
        distance = np.linalg.norm(normalized_lda_features_new_description[0] - centroid)
        distances.append(distance)

    closest_cluster = np.argmin(distances)
    #print(f"The new business description belongs to cluster {closest_cluster}")
    #merged_list.append(closest_cluster)
    a = np.delete(distances, closest_cluster)
    closest_cluster2 = np.argmin(a)
    lab2.append(closest_cluster2)

    # Get the descriptions in the predicted cluster
    cluster_descriptions = []
    for i in range(len(data)):
        if cluster_labels[i] == predicted_cluster:
            cluster_descriptions.append(data[i])
    print("predicted_cluster: " ,predicted_cluster)
    processed_new_description = preprocess_text2222(cluster_descriptions)
    bow_new_description = dictionary.doc2bow(processed_new_description)
    lda_new_description = lda[bow_new_description]
    lda_features_new_description = np.zeros((1, num_topics))

    for topic_id, topic_prob in lda_new_description:
        lda_features_new_description[0, topic_id] = topic_prob

    normalized_lda_features_new_description33 = scaler.transform(lda_features_new_description)
    
    # Compute the cosine similarity between the new description and the descriptions in the predicted cluster
    similarity_scores = cosine_similarity(normalized_lda_features_new_description, normalized_lda_features_new_description33)
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

    ########## the second 
    print("")
    print("the second")
    print("")
    # Get the descriptions in the predicted cluster
    cluster_descriptions = []
    for i in range(len(data)):
        if cluster_labels[i] == closest_cluster2:
            cluster_descriptions.append(data[i])
    print("predicted_cluster: " ,closest_cluster2)

    
    processed_new_description = preprocess_text2222(closest_cluster2)
    bow_new_description = dictionary.doc2bow(processed_new_description)
    lda_new_description = lda[bow_new_description]
    lda_features_new_description = np.zeros((1, num_topics))

    for topic_id, topic_prob in lda_new_description:
        lda_features_new_description[0, topic_id] = topic_prob

    normalized_lda_features_new_description66 = scaler.transform(lda_features_new_description)
    

    # Compute the cosine similarity between the new description and the descriptions in the predicted cluster
    similarity_scores = cosine_similarity(normalized_lda_features_new_description, normalized_lda_features_new_description66)
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
    # Print the maximum similarity score
    #print("Max similarity score:", np.max(similarity_scores))
    # Print the similarity scores
    print("Similarity scores:")
    print("")
    '''
    for i in range(3): #len(cluster_descriptions)
        print(f"{cluster_descriptions[i]}: {similarity_scores[0][i]}")
    print("******")
    '''
    max_scores_indices = np.argsort(-similarity_scores)
    #print("Top 3 similarity scores:")
    #for i in range(3):
    #    print(f"{cluster_descriptions[max_scores_indices[0][i]]}: {similarity_scores[0][max_scores_indices[0][i]]}")
    
    print("Predicted closest_cluster:", closest_cluster)
    print("Predicted closest_cluster2:", closest_cluster2)
    #print("Second closest cluster:", second_closest_cluster)


print(" ")


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


# Add cluster labels to the dataframe
business_df['cluster'] = kmeans.labels_

print("Print the size of each cluster")
# Print the size of each cluster
print(list(business_df.groupby('cluster').size()))    #[10, 3, 28, 52, 100, 
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




print("Silhouette score:", silhouette)
print(" ")

# Count the occurrences of each number
counts1 = Counter(labels)
# Print the counts
labbb=[]
for number, count in counts1.items():
    #print(f"Number {number} appears {count} times")
    labbb.append(count)

print("Print the size of each cluster of the input data" ,labbb)
print(" ")

print("lab1 : " , lab1)

# Count the occurrences of each number
counts1 = Counter(lab1)
# Print the counts
labbb=[]
for number, count in counts1.items():
    #print(f"Number {number} appears {count} times")
    labbb.append(count)

print("Print the size of each cluster of the new data one cluster" ,labbb)
print(" ")
print("lab2 : " , lab2)

# Count the occurrences of each number
counts1 = Counter(lab2)
# Print the counts
labbb=[]
for number, count in counts1.items():
    #print(f"Number {number} appears {count} times")
    labbb.append(count)

print("Print the size of each cluster of the new data tow cluster" ,labbb)
print(" ")



# assuming lda_model is your trained LdaModel
num_topics = lda.num_topics  # number of topics
num_words = 10  # number of words you want to display per topic

for i in range(num_topics):
    top_words = lda.show_topic(i, num_words)
    print(f"Top words for topic #{i}:")
    print(", ".join([word for word, _ in top_words]))

