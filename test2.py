from bertopic import BERTopic

# Load your pre-trained BERTopic model
topic_model = BERTopic.load("MaartenGr/BERTopic_ArXiv")

documents = [
    "Natural language processing is a branch of artificial intelligence.",
    "BERTopic is a powerful library for topic modeling with BERT embeddings.",
    "Topic modeling is useful for discovering underlying themes in text data.",
    "The BERTopic library provides an easy-to-use interface for topic modeling.",
    "Machine learning algorithms are used for various tasks in natural language processing.",
    "BERT embeddings capture contextual information from text.",
    "Clustering algorithms can group similar documents together.",
    "The cosine similarity metric is often used in text analysis.",
    "Document clustering helps in organizing large collections of text data.",
    "Understanding topics in text can aid in information retrieval tasks."
]

# Adjust the number of topics to match the pre-trained model
num_topics = len(topic_model.get_topic_info())

# 2. Create Topic Model
topics, _ = topic_model.transform(documents)

# 3. Get Topic Information
topic_info = topic_model.get_topic_info()

# 4. Interpret Results
print(topic_info)

# Print the topic names for each document
# Print the main topic names for each document
for i, topic in enumerate(topics):
    topic_label = topic_info.loc[topic_info['Topic'] == topic]['Name'].values[0]
    print(f"Document {i}: Topic {topic_label}")
