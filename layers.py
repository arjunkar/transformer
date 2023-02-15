"""
Defines various layers necessary for implementing the 
encoder and decoder phases of a Transformer.
Both phases have the same hyperparameters for simplicity.
Intended for use in PyTorch's WikiText2 tutorial:
https://pytorch.org/tutorials/beginner/transformer_tutorial.html

With the source mask enabled as in the provided training loop,
but without a target or memory mask, the model can achieve 
a test loss of 0.89 and a test perplexity of 2.43 on the WikiText2 tutorial.
=========================================================================================
| End of training | test loss  0.89 | test ppl     2.43
=========================================================================================
"""

import math
import torch
from torch import nn

##### ZEROTH LEVEL CLASSES ####################################################################################################

class MyTransformer(nn.Module):
    def __init__(self,vocab,d_model,n_heads,d_hidden,n_layers,dropout=0.5) -> None:
        super().__init__()
        self.embedder = Embedder(vocab,d_model,dropout)
        self.encoder = EncoderStack(d_model,n_heads,d_hidden,n_layers,dropout)
        self.decoder = DecoderStack(d_model,n_heads,d_hidden,n_layers,dropout)
        self.logits = LinearLayer(d_model,vocab)

    def forward(self,input,src_mask=None,tgt_mask=None,mem_mask=None):
        x = self.embedder(input)
        enc_output = self.encoder(x,src_mask)
        x = self.decoder(x,enc_output,tgt_mask,mem_mask)
        return self.logits(x)
        # Reusing input sequence as target for training purposes.
        # The call to the model in PyTorch's tutorial must be modified
        # to deal with the additional positional mask arguments here.


# An example of the type of mask accepted by the transformer.
# This is an "additive" mask meant to be added to the tensor to be masked
# prior to a softmax application.
# Taken from the referenced PyTorch tutorial.

def generate_square_subsequent_mask(sz: int):
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)



##### FIRST LEVEL CLASSES ####################################################################################################

class Embedder(nn.Module):
    """
    The transformer input in the WikiText2 tutorial is tokenized so that
    the outermost input is [seq_len,batch_dim] and consists of numbers from
    0 <= num < vocab.  Each of these numbers needs to be embedded in a word vector.
    """
    def __init__(self,vocab,d_model,dropout=0.5) -> None:
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        #self.embed = nn.Parameter(torch.randn(vocab,d_model))
        self.embed = nn.Embedding(vocab,d_model)
        self.pos = Positional(d_model)

    def forward(self,input):
        x = self.embed(input)
        x = self.pos(x)
        return self.dropout(x)

class DecoderStack(nn.Module):
    def __init__(self,d_model,n_heads,d_hidden,n_layers,dropout=0.5):
        super().__init__()
        self.decode_layers = nn.ModuleList(
            [DecodeLayer(d_model,n_heads,d_hidden,dropout) for _ in range(n_layers)]
            )

    def forward(self,input,memory,ipt_mask=None,mem_mask=None):
        x = input
        for layer in self.decode_layers:
            x = layer(x,memory,ipt_mask,mem_mask)
        return x

class EncoderStack(nn.Module):
    def __init__(self,d_model,n_heads,d_hidden,n_layers,dropout=0.5):
        super().__init__()
        self.encode_layers = nn.ModuleList(
            [EncodeLayer(d_model,n_heads,d_hidden,dropout) for _ in range(n_layers)]
            )

    def forward(self,input,ipt_mask=None):
        x = input
        for layer in self.encode_layers:
            x = layer(x,ipt_mask)
        return x



##### SECOND LEVEL CLASSES ####################################################################################################


class Positional(nn.Module):
    def __init__(self,d_model,max_len=5000) -> None:
        super().__init__()
        PE = torch.unsqueeze(
            torch.tensor(
                [
                [math.sin(p / (10000.**(i/d_model))) if i%2 == 0 
                    else math.cos(p / (10000.**((i-1)/d_model))) for i in range(d_model)]
                for p in range(max_len)
                ],
                requires_grad=False), # PyTorch should treat PE as constant
            dim = 1)
        # Broadcast positional encoding addition over batch dimension 1
        self.register_buffer('PE',PE)
    
    def forward(self,input):
        # shape of input: [seq_len, batch_dim, d_model]
        # Processing one position (seq_len) of each input in batch at a given time.
        return input + self.PE[:input.size(dim=0)]
        # Cut PE tensor along seq_len dimension to match input -- fails if over max_len


class DecodeLayer(nn.Module):
    def __init__(self,d_model,n_heads,d_hidden,dropout=0.5):
        super().__init__()
        # Self-attention block
        self.self_attn0 = AttentionLayer(d_model, n_heads)
        self.dropout0 = nn.Dropout(dropout)
        self.norm0 = LayerNormalization(d_model)
        # Encoder-decoder attention block
        self.self_attn1 = AttentionLayer(d_model, n_heads)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = LayerNormalization(d_model)
        # Feedforward block
        self.linear1 = LinearLayer(d_model,d_hidden)
        self.activation = nn.functional.relu
        self.dropout = nn.Dropout(dropout)
        self.linear2 = LinearLayer(d_hidden,d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = LayerNormalization(d_model)
        
    def forward(self,input,memory,ipt_mask=None,mem_mask=None):
        x = input
        x = self.norm0(x + self.dropout0(self.self_attn0(x,x,x,ipt_mask)))
        x = self.norm1(x + self.dropout1(self.self_attn1(x,memory,memory,mem_mask)))
        x = self.norm2(x + self.ffwd(x))
        return x

    # feed forward function
    def ffwd(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)



class EncodeLayer(nn.Module):
    def __init__(self,d_model,n_heads,d_hidden,dropout=0.5):
        super().__init__()
        # Attention block
        self.self_attn = AttentionLayer(d_model, n_heads)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = LayerNormalization(d_model)
        # Feedforward block
        self.linear1 = LinearLayer(d_model,d_hidden)
        self.activation = nn.functional.relu
        self.dropout = nn.Dropout(dropout)
        self.linear2 = LinearLayer(d_hidden,d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = LayerNormalization(d_model)
        
    def forward(self,input,ipt_mask=None):
        x = input
        x = self.norm1(x + self.dropout1(self.self_attn(x,x,x,ipt_mask)))
        x = self.norm2(x + self.ffwd(x))
        return x

    # feed forward function
    def ffwd(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)



##### THIRD LEVEL CLASSES ####################################################################################################



class AttentionLayer(nn.Module):
    def __init__(self,d_model,n_heads) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.WQuery = nn.ModuleList(
            [LinearLayer(d_model,self.d_head) for _ in range(n_heads)]
            )
        self.WKey = nn.ModuleList(
            [LinearLayer(d_model,self.d_head) for _ in range(n_heads)]
            )
        self.WValue = nn.ModuleList(
            [LinearLayer(d_model,self.d_head) for _ in range(n_heads)]
            )
        self.WOut = LinearLayer(n_heads*(self.d_head),d_model)

    def forward(self,query,key,value,mask=None):
        query = torch.transpose(query,dim0=0,dim1=1)
        key = torch.transpose(key,dim0=0,dim1=1)
        value = torch.transpose(value,dim0=0,dim1=1)
        """
        Transpose from [seq_len,batch_dim,d_model] to [batch_dim,seq_len,d_model]
        This facilitates multiplication by torch.matmul as we want to take inner
        products between queries and keys from the same batch index but different
        sequence position and feature index.
        """
        queries = [self.WQuery[i](query) for i in range(self.n_heads)]
        keys = [self.WKey[i](key) for i in range(self.n_heads)]
        values = [self.WValue[i](value) for i in range(self.n_heads)]
        activations = self.attention(queries, keys, values, mask)
        return torch.transpose(
            self.WOut(torch.cat(activations,dim=-1)),
            dim0=0,dim1=1)
        # Transpose back to [seq_len,batch_dim,d_model]


    def attention(self,queries,keys,values,mask=None):
        preactivations = [torch.matmul(
              queries[i],torch.transpose(keys[i],dim0=-1,dim1=-2)
            ) / (self.d_head**0.5) for i in range(self.n_heads)]
        activations = [torch.matmul(
                nn.functional.softmax(
                    preactivations[i] + (mask if mask is not None else 0.), dim=-1
                ), values[i]
            ) for i in range(self.n_heads)]
        """
        The mask must be passed as a [seq_len,seq_len] matrix 
        with float('-inf') in the reset positions and zero elsewhere.
        This allows a broadcasted addition to reset the preactivations
        in every batch element.
        """
        return activations
            



class LinearLayer(nn.Module):
    def __init__(self,dim_in,dim_out):
        super().__init__()
        self.dim_in = dim_in
        self.W = nn.Parameter(torch.empty(dim_in,dim_out))
        # Row (not column) of input matrix is one input vector so W is transposed from W.x + b
        self.b = nn.Parameter(torch.empty(dim_out))

        self.initialize()

    def initialize(self):
        nn.init.uniform_(self.W, a=-1.0/self.dim_in**0.5, b=1.0/self.dim_in**0.5)
        nn.init.uniform_(self.b, a=-1.0/self.dim_in**0.5, b=1.0/self.dim_in**0.5)
        # Initialization which preserves the large-size limit is crucial.
        # Perplexity on WikiText2 can increase by orders of magnitude if
        # the linear layers are not initialized properly.

    def forward(self,input):
        return torch.matmul(input,self.W)+self.b
        # Broadcast addition of b over batch input



class LayerNormalization(nn.Module):
    def __init__(self,d_model,eps=1e-5) -> None:
        super().__init__()
        self.eps = eps # Regularization to protect against small standard deviation
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))


    def forward(self,input):
        means = torch.mean(input,dim=-1,keepdim=True)
        vars = torch.var(input,dim=-1,keepdim=True,unbiased=False)
        # Dimension -1 ensures we compute statistics along the feature dimension
        # By convention, the sequence and batch dimensions come first
        return self.gamma * (input-means)/((vars+self.eps)**0.5) + self.beta
        # The additional optimizable parameters are crucial for performance on
        # WikiText2.




##### TESTING ####################################################################################################
"""
This section enables Tensorboard analysis of the graph 
and performance of the model on a fake input.
"""


"""
vocab = 15
d_model = 6
n_heads = 2
d_hidden = 4
n_layers = 2
batch_size = 3
seq_len = 4


from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('transformer/runs/transformer_test_1')


myModel = MyTransformer(vocab,d_model,n_heads,d_hidden,n_layers)
testInput = torch.ones(size=(seq_len,batch_size),dtype=int)

writer.add_graph(myModel, testInput)
writer.close()

print(myModel(testInput))
"""

