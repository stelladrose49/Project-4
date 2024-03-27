from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import numpy as np
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
import string
from nltk.corpus import stopwords

# 创建 Flask 应用实例
app = Flask(__name__)
# 启用 CORS
CORS(app, resources={r"/*": {"origins": "*"}})

# 加载事先训练好的模型、列名和标准化器
model = pickle.load(open('best_rf_model.pkl', 'rb'))
model_columns = pickle.load(open('model_columns.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))  # 加载标准化器

# 加载预训练的Word2Vec模型
word2vec = Word2Vec.load("word2vec.model")

# 定义SentimentTopicModel类
class SentimentTopicModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, sentiment_classes, topic_classes):
        super(SentimentTopicModel, self).__init__()
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_sentiment = nn.Linear(hidden_dim, sentiment_classes)
        self.fc_topic = nn.Linear(hidden_dim, topic_classes)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        sentiment = self.fc_sentiment(x)
        topic = self.fc_topic(x)
        return sentiment, topic

# 设置参数值
embedding_dim = 100
hidden_dim = 200
sentiment_classes = 5
topic_classes = 5
sentiment_map = {'Joy': 0, 'Surprise': 1, 'Neutral': 2, 'Disappointment': 3, 'Anger': 4}

# 加载预训练的情感主题预测模型
sentiment_model = SentimentTopicModel(embedding_dim, hidden_dim, sentiment_classes, topic_classes)
sentiment_model.load_state_dict(torch.load("sentiment_model.pth"))
sentiment_model.eval()

sentiment_labels = {v: k for k, v in sentiment_map.items()}
topic_labels = [
    "Overall Phone Quality",
    "SIM Card and Carrier",
    "Battery and Charging",
    "Price and Value",
    "General Phone Features"
]

# 定义预测的路由,只接受 POST 请求
@app.route('/predict', methods=['POST'])
def predict():
    # 获取通过 POST 请求发送的 JSON 数据
    data = request.get_json()
    print("Received data:", data)

    # 将 JSON 数据转换成 pandas DataFrame
    new_phone = pd.DataFrame({
        'Brand Name': [data['Brand Name']],
        'Chipset': [data['Chipset']],
        'Display': [data['Display']],
        'Battery': [data['Battery']],
        'Storage (GB)': [data['Storage (GB)']],
        'RAM (GB)': [data['RAM (GB)']],
        'Camera Max MP': [data['Camera Max MP']]
    })

    brand_one_hot = pd.get_dummies(new_phone['Brand Name'], prefix='Brand Name')
    chipset_one_hot = pd.get_dummies(new_phone['Chipset'], prefix='Chipset')

    new_phone_preprocessed = pd.concat([new_phone.drop(['Brand Name', 'Chipset'], axis=1), brand_one_hot, chipset_one_hot], axis=1)
    new_phone_preprocessed = new_phone_preprocessed.reindex(columns=model_columns, fill_value=0)

    numeric_features = ['Display', 'Battery', 'Storage (GB)', 'RAM (GB)', 'Camera Max MP']
    new_phone_preprocessed[numeric_features] = scaler.transform(new_phone_preprocessed[numeric_features])

    # 使用加载的模型进行预测
    price_prediction = model.predict(new_phone_preprocessed)
    print("Predicted price:", price_prediction[0])

    # 将预测结果包装在 JSON 中并返回
    return jsonify({'predicted_price': price_prediction[0]})

# 文本预处理函数
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = word_tokenize(text)
    words = [word for word in words if word not in stop_words]
    return words

# 将文本转换为词嵌入向量
def text_to_embedding(text, word2vec):
    words = word_tokenize(text)
    vectors = [word2vec.wv[word] for word in words if word in word2vec.wv]
    if len(vectors) == 0:
        return np.zeros(100)
    return np.mean(vectors, axis=0)

# 定义情感和主题预测的路由,只接受 POST 请求
@app.route('/predict_review', methods=['POST'])
def predict_review():
    # 获取通过 POST 请求发送的 JSON 数据
    data = request.get_json()
    print("Received data:", data)

    # 将评论按行分割
    reviews = data['review'].split('\n')

    predicted_sentiments = []
    predicted_topics = []

    for review in reviews:
        # 预处理评论
        processed_review = preprocess_text(review)

        # 将评论转换为词嵌入向量
        review_embedding = text_to_embedding(' '.join(processed_review), word2vec)

        # 使用训练好的模型进行预测
        with torch.no_grad():
            sentiment_output, topic_output = sentiment_model(torch.from_numpy(review_embedding).float().unsqueeze(0))
            _, sentiment_prediction = torch.max(sentiment_output, 1)
            _, topic_prediction = torch.max(topic_output, 1)

        predicted_sentiments.append(sentiment_labels[sentiment_prediction.item()])
        predicted_topics.append(topic_labels[topic_prediction.item()])

    # 将预测结果包装在 JSON 中并返回
    return jsonify({'predicted_sentiments': predicted_sentiments, 'predicted_topics': predicted_topics})

# 如果这个脚本是直接运行的,启动 Flask 应用
# 如果是通过 WSGI 服务器运行的,这部分将被忽略
if __name__ == '__main__':
    app.run(debug=True)  # 在调试模式下运行 Flask 应用