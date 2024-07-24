import torch
import torch.nn as nn
import torchvision.models as models



class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        for param in resnet.parameters():
            param.requires_grad_(False)
            
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Sequential(
            nn.Linear(resnet.fc.in_features, embed_size),
            nn.ReLU()
        )

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        
        return features



class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=2, dropout=0.5):
        super(DecoderRNN, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers

        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)
        
        # Initialize weights
        self.init_weights()

    def init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def forward(self, features, captions):
        captions = captions[:, :-1]  # Remove <end> token
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), dim=1)
        
        lstm_out, _ = self.lstm(embeddings)
        lstm_out = self.dropout(lstm_out)
        outputs = self.fc(lstm_out)
        
        return outputs

    def sample(self, inputs, states=None, max_len=30):
        res = []
        for i in range(max_len):
            lstm_out, states = self.lstm(inputs, states)
            lstm_out = self.dropout(lstm_out)
            outputs = self.fc(lstm_out.squeeze(1))
            _, predicted_idx = outputs.max(1)
            
            res.append(predicted_idx.item())
            
            # if predicted_idx.item() == tokenizer.sep_token_id:  # <SEP> token
            if predicted_idx.item() == 102: 
                break
                
            inputs = self.embed(predicted_idx).unsqueeze(1)
        
        return res