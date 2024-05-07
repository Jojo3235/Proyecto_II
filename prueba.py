from bertopic import BERTopic
topic_model = BERTopic.load("MaartenGr/BERTopic_ArXiv")

topic_model.get_topic_info()
