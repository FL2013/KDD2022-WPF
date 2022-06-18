import paddle
import paddle.nn as nn 
import math
import paddle.tensor as Tensor

class PositionalEncoder(nn.Layer):

    def __init__(self, dropout: float = 0.1, max_seq_len: int = 5000, d_model: int = 512):

        """
        Args:
            dropout: the dropout rate
            max_seq_len: the maximum length of the input sequences
            d_model: The dimension of the output of sub-layers in the model 
        """

        super().__init__()

        self.d_model = d_model
        
        self.dropout = nn.Dropout(p=dropout)

        # Create constant positional encoding matrix with values 
        # dependent on position and i
        position = paddle.arange(max_seq_len).unsqueeze(1).astype('float32')
        
        exp_input = paddle.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        
        div_term = paddle.exp(exp_input) # Returns a new tensor with the exponential of the elements of exp_input
        
        self.pe = paddle.zeros((max_seq_len, d_model))
        
        self.pe[:, 0::2] = paddle.sin(position * div_term)
        
        self.pe[:, 1::2] = paddle.cos(position * div_term) # torch.Size([target_seq_len, dim_val])

        self.pe = self.pe.unsqueeze(0) # torch.Size([target_seq_len, input_size, dim_val])
        self.pe = paddle.transpose(self.pe, [1,0,2])
        
        
    def forward(self, x:Tensor) -> Tensor :
        """
        Args:
            x: Tensor, shape [batch_size, enc_seq_len, dim_val]
        """
        
        add = self.pe.squeeze(1)

        x = x + add

        return self.dropout(x)