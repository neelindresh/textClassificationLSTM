import torch   
import textClassification as textModel
#handling text data
import warnings
from torchtext import data 

warnings.filterwarnings("ignore")
TEXT = data.Field(tokenize='spacy',batch_first=True,include_lengths=True)
LABEL = data.LabelField(dtype = torch.float,batch_first=True)
fields = [('text',TEXT),('label', LABEL)]

training_data=data.TabularDataset(path = 'classfication_data.csv',
                                 format = 'csv',
                                 fields = fields,
                                skip_header = True)

train_data, valid_data = training_data.split(split_ratio=0.9,)

TEXT.build_vocab(train_data,min_freq=3,vectors = "glove.6B.100d")  
LABEL.build_vocab(train_data)

BATCH_SIZE = 64

#Load an iterator
train_iterator, valid_iterator = data.BucketIterator.splits(
    (train_data, valid_data), 
    batch_size = BATCH_SIZE,
    sort_key = lambda x: len(x.text),
    sort_within_batch=True
    )

size_of_vocab = len(TEXT.vocab)
embedding_dim = 100
num_hidden_nodes = 128
num_layers = 2
N_EPOCHS = 20
best_valid_loss = float('inf')




model=textModel.Classifier(size_of_vocab,embedding_dim,num_hidden_nodes,num_layers)
pretrained_embeddings = TEXT.vocab.vectors
model.embedding.weight.data.copy_(pretrained_embeddings)

optimizer = torch.optim.Adam(model.parameters())
criterion = torch.nn.BCELoss()



for epoch in range(N_EPOCHS):
     
    #train the model
    train_loss, train_acc = textModel.train(model, train_iterator, optimizer, criterion)
    
    #evaluate the model
    valid_loss, valid_acc = textModel.evaluate(model, valid_iterator, criterion)
    
    print("EPOCH:",epoch)
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}% |Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')
    