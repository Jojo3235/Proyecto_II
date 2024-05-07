import pandas as pd
from bertopic import BERTopic

# Load your pre-trained BERTopic model
topic_model = BERTopic.load("MaartenGr/BERTopic_ArXiv")

# Read the CSV file
data = pd.read_csv("tweetspersonas.csv")

# Extract the text data and the corresponding user from the CSV file
documents = data["full_text"].tolist()
users = data["user"].tolist()

# Adjust the number of topics to match the pre-trained model
num_topics = len(topic_model.get_topic_info())

# 2. Create Topic Model
topics, _ = topic_model.transform(documents)

# 3. Get Topic Information
topic_info = topic_model.get_topic_info()

# Create a list to store dictionaries of user and classification
result_list = []

# Iterate through each user and their corresponding tweet's topic
for user, topic in zip(users, topics):
    topic_label = topic_info.loc[topic_info['Topic'] == topic]['Name'].values[0]
    result_list.append({"user": user, "classification": topic_label})

# Convert the list of dictionaries to a DataFrame
result_df = pd.DataFrame(result_list)

# Save the results to a new CSV file
result_df.to_csv("user_classifications.csv", index=False)

# Print the first few rows of the result DataFrame
print(result_df.head())
