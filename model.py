import torch
import torch.nn as nn
import torchvision.models as models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=3):
        ''' Initialize the layers of this model.'''
        super(DecoderRNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        # embedding layer that turns words into a vector of a specified size
        self.embeds = nn.Embedding(vocab_size, embed_size)

        # the LSTM takes embedded word vectors (of a specified size) as inputs 
        # and outputs hidden states of size hidden_size
        self.lstm = nn.LSTM(input_size=embed_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)

        # the linear layer that maps the hidden state output dimension 
        # to the number of tags we want as output, tagset_size (in this case this is 3 tags)
        self.hidden2tag = nn.Linear(hidden_size, vocab_size)
        
        
        
    def init_hidden(self,batch_size):
        ''' At the start of training, we need to initialize a hidden state;'''
        # The axes dimensions are (n_layers, batch_size, hidden_dim)
        return (torch.zeros((self.num_layers, batch_size, self.hidden_size), device=device), 
                torch.zeros((self.num_layers, batch_size, self.hidden_size), device=device))
    
    def forward(self, features, captions):
        ''' Define the feedforward behavior of the model.'''
        
        # Initializing the hidden states for each forward iteration 
        hidden = self.init_hidden(features.shape[0])
        
        # create embedded word vectors for each word in a sentence
        embeds = self.embeds(captions[:,:-1])
        
        # concatenate the fe#ature and caption embeds
        inputs = torch.cat((features.unsqueeze(1),embeds),1)
        
        # get the output and hidden state by passing the lstm over our word embeddings and hidden states
        lstm_out, _ = self.lstm(inputs, hidden)
        
        # get the scores for the most likely tag for a word
        tag_outputs = self.hidden2tag(lstm_out)
        #tag_scores = F.log_softmax(tag_outputs, dim=1)
        
        return tag_outputs

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        ids = []
        states = (torch.randn((self.num_layers, 1, self.hidden_size), device=device), 
                  torch.randn((self.num_layers, 1, self.hidden_size), device=device))
        for i in range(max_len):
            lstm_out, states = self.lstm(inputs, states)
            outputs = self.hidden2tag(lstm_out.squeeze(1))
            _,wordid = outputs.max(1)
            ids.append(wordid.item())
            inputs = self.embeds(wordid) 
            inputs = inputs.unsqueeze(1)
        return ids