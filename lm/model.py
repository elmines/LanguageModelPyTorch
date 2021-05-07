# 3rd Party
import torch

class LanguageModel(torch.nn.Module):
    def __init__(self, in_tokens : int, vocab_size : int, embedding_size : int, activ = None):
        super(LanguageModel, self).__init__()
        self._activation     = activ if activ else torch.nn.Tanh()
        self._in_tokens      = in_tokens
        self._embedding_size = embedding_size
        self.embedding       = torch.nn.Embedding(vocab_size, embedding_size)
        self.linear          = torch.nn.Linear(in_tokens*embedding_size, vocab_size)
    def forward(self, x : torch.Tensor):
        x = self.embedding(x)
        x = x.view( *x.shape[:-2], self._in_tokens * self._embedding_size)
        x = self._activation(x)
        x = self.linear(x)
        x = torch.softmax(x, dim=-1)
        return x
