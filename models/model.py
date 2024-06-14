import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoConfig
import torch
import os

base_dir = os.path.dirname(os.path.abspath(__file__))

# Load the tokenizer and models
tokenizer = AutoTokenizer.from_pretrained(os.path.join(base_dir, "tokenizer"))
config = AutoConfig.from_pretrained("crypter70/IndoBERT-Sentiment-Analysis")
model = AutoModel.from_pretrained(os.path.join(base_dir, "indobert_model"), config=config)

class BertForMultiTaskSequenceClassification(nn.Module):
    def __init__(self, model):
        super(BertForMultiTaskSequenceClassification, self).__init__()
        self.bert = model
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state.mean(dim=1)
        pooled_output = self.dropout(pooled_output)
        return pooled_output

embedding_model = BertForMultiTaskSequenceClassification(model)

class Attention(nn.Module):
    def __init__(self, attention_size):
        super(Attention, self).__init__()
        self.attention_weights = nn.Linear(attention_size, 1, bias=False)

    def forward(self, lstm_output):
        attention_scores = self.attention_weights(lstm_output).squeeze(-1)
        attention_weights = torch.softmax(attention_scores, dim=1)
        weighted_sum = torch.sum(lstm_output * attention_weights.unsqueeze(-1), dim=1)
        return weighted_sum, attention_weights

class ABSA_LSTM_CNN(nn.Module):
    def __init__(self, embedding_model, input_size=768, hidden_size=256, num_layers=1, num_classes_sentiment=4,num_classes_confidence=4):
        super(ABSA_LSTM_CNN, self).__init__()
        self.embedding_model = embedding_model
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=256, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(256, hidden_size, num_layers, batch_first=True, dropout=0.2, bidirectional=True)
        self.attention = Attention(hidden_size * 2)

        self.fc1 = nn.Linear(hidden_size * 2, 128)
        self.dropout1 = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(256, 128)

        self.fc_sentiment = nn.Linear(128, num_classes_sentiment)
        self.fc_confidence = nn.Linear(128, num_classes_confidence)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        embeddings = self.embedding_model(input_ids, attention_mask)
        embeddings = embeddings.unsqueeze(1).repeat(1, input_ids.size(1), 1)

        x = embeddings.permute(0, 2, 1)
        x = self.relu(self.conv1(x))
        x = x.permute(0, 2, 1)

        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(input_ids.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(input_ids.device)
        out, _ = self.lstm(x, (h0, c0))

        attn_output, _ = self.attention(out)

        out = self.fc1(attn_output)
        out = self.relu(out)
        out = self.dropout1(out)
#         out = self.fc2(out)
        out = self.relu(out)

        sentiment = self.fc_sentiment(out)
        sentiment = self.tanh(sentiment)

        confidence = self.fc_confidence(out)
        confidence = self.sigmoid(confidence)

        return sentiment, confidence

no_grad = torch.no_grad()
model_absa = ABSA_LSTM_CNN(embedding_model)
model_absa.load_state_dict(torch.load(os.path.join(base_dir, "FastBack")), strict=False)