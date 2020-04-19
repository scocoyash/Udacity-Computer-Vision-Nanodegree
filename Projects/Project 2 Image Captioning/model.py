import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        # Embedding Vector
        self.embed = nn.Embedding(vocab_size, embed_size)
        # lstm
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        # fully connected
        self.fc = nn.Linear(hidden_size, vocab_size)
        
        self.init_weights()
        
    def init_weights(self):
        torch.nn.init.xavier_uniform_(self.fc.weight)
        torch.nn.init.xavier_uniform_(self.embed.weight)
        
    def forward(self, features, captions):
        # generate embeddings from caption
        captions = captions[:,:-1]
        captions = self.embed(captions)
        
        # concat the features and captions
        inputs = torch.cat((features.unsqueeze(1),captions), 1)
        
        # pass through lstm and fc
        outputs, _ = self.lstm(inputs)
        outputs = self.fc(outputs)
        
        return outputs

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        outputs = []
        output_length = 0
        
        while (output_length != max_len + 1):
            output, states = self.lstm(inputs, states)
            output = self.fc(output.squeeze(1))
            _, predicted_index = torch.max(output, 1)
            # because numpy doesn't support GPU
            outputs.append(predicted_index.cpu().numpy()[0].item())
            
            # <end> token in case caption is less than 20 
            if(predicted_index == 1):
                break
            
            # next iteration inputs
            inputs = self.embed(predicted_index)   
            inputs = inputs.unsqueeze(1)
            
            output_length += 1
        
        return outputs