class Vocab():
    def __init__(self):
        self.word2index={
        }
        self.index2word={
        }
        self.word2count={
        }
        self.nwords=0
        self.maxwords=0

    def add_word(self,sentence):
        for i in sentence:
            if i in self.word2index:
                self.word2count[i]=self.word2count[i]+1
            else:
                self.word2index[i]=self.nwords+1
                self.index2word[self.nwords+1]=i
                self.word2count[i]=1
                self.nwords+=1
        if self.maxwords<len(sentence):
            self.maxwords=len(sentence)


