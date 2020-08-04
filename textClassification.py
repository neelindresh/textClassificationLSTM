import torch

class Classifier(torch.nn.Module):
    def __init__(self,vocab_size,embedding_size,hidden_size,num_layers):
        super(Classifier, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size=embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding=torch.nn.Embedding(vocab_size,embedding_size)
        self.lstm=torch.nn.LSTM(embedding_size,hidden_size,num_layers,batch_first=True)
        self.fc=torch.nn.Linear(hidden_size,1)
        self.ac=torch.nn.Sigmoid()
    def forward(self,x,batch_size=64):
        embed=self.embedding(x)
        out,(hidden,cell)=self.lstm(embed)
        #print(self.num_layers,self.hidden_size)
        #print(hidden[-1].unsqueeze(0).size())

        #Interested in Last Layer of LSTM
        hidden=hidden[-1].unsqueeze(0)
        out=self.fc(hidden)
        out=self.ac(out)
        return out

def binary_accuracy(preds, y):
    #round predictions to the closest integer
    rounded_preds = torch.round(preds)
    
    correct = (rounded_preds == y).float() 
    acc = correct.sum() / len(correct)
    return acc
def train(model, iterator, optimizer, criterion):
    
    #initialize every epoch 
    epoch_loss = 0
    epoch_acc = 0
    
    #set the model in training phase
    model.train()  
    
    for batch in iterator:
        
        #resets the gradients after every batch
        optimizer.zero_grad()   
        
        #retrieve text and no. of words
        text, text_lengths = batch.text   
        
        #convert to 1D tensor
        predictions = model(text).squeeze()  
        
        #compute the loss
        loss = criterion(predictions, batch.label)        
        
        #compute the binary accuracy
        acc = binary_accuracy(predictions, batch.label)   
        
        #backpropage the loss and compute the gradients
        loss.backward()       
        
        #update the weights
        optimizer.step()      
        
        #loss and accuracy
        epoch_loss += loss.item()  
        epoch_acc += acc.item()    
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)
def evaluate(model, iterator, criterion):
    
    #initialize every epoch
    epoch_loss = 0
    epoch_acc = 0

    #deactivating dropout layers
    model.eval()
    
    #deactivates autograd
    with torch.no_grad():
    
        for batch in iterator:
        
            #retrieve text and no. of words
            text, text_lengths = batch.text
            
            #convert to 1d tensor
            predictions = model(text,).squeeze()
            
            #compute loss and accuracy
            loss = criterion(predictions, batch.label)
            acc = binary_accuracy(predictions, batch.label)
            
            #keep track of loss and accuracy
            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)
