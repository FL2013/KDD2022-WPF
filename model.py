import paddle
import paddle.nn as nn
import paddle.tensor as Tensor
import positional_encoder as pe

class TimeSeriesTransformer(nn.Layer):
    def __init__(self, 
        input_size: int,
        enc_seq_len: int,
        dec_seq_len: int,
        max_seq_len: int,
        out_seq_len: int,
        dim_val: int=512,  
        n_encoder_layers: int=4,
        n_decoder_layers: int=4,
        n_heads: int=8,
        dropout_encoder: float=0.2, 
        dropout_decoder: float=0.2,
        dropout_pos_enc: float=0.2,
        dim_feedforward_encoder: int=512,
        dim_feedforward_decoder: int=512,
        ): 

        """
        Args:
            input_size: int, number of input variables. 
            dec_seq_len: int, the length of the input sequence fed to the decoder
            max_seq_len: int, length of the longest sequence the model will 
                         receive. Used in positional encoding. 
            out_seq_len: int, the length of the model's output 
            dim_val: int, aka d_model. All sub-layers in the model produce 
                     outputs of dimension dim_val
            n_encoder_layers: int, number of stacked encoder layers in the encoder
            n_decoder_layers: int, number of stacked encoder layers in the decoder
            n_heads: int, the number of attention heads
            dropout_encoder: float, the dropout rate of the encoder
            dropout_decoder: float, the dropout rate of the decoder
            dropout_pos_enc: float, the dropout rate of the positional encoder
            dim_feedforward_encoder: int, number of neurons in the linear layer 
                                     of the encoder
            dim_feedforward_decoder: int, number of neurons in the linear layer 
                                     of the decoder
        """

        super().__init__() 
        self.enc_seq_len = enc_seq_len
        self.dec_seq_len = dec_seq_len
        self.out_seq_len = out_seq_len

        # Creating the three linear layers needed for the model
        self.encoder_input_layer = nn.Linear(
            in_features=input_size, 
            out_features=dim_val 
            )

        self.decoder_input_layer = nn.Linear(
            in_features=input_size, 
            out_features=dim_val 
            )  

        self.linear_mapping = nn.Linear(
            in_features=dim_val,
            out_features=1
            )

        # Create positional encoder
        self.positional_encoding_layer = pe.PositionalEncoder(
            d_model=dim_val,
            dropout=dropout_pos_enc,
            max_seq_len=max_seq_len
            )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_val, 
            nhead=n_heads,
            dim_feedforward=dim_feedforward_encoder,
            dropout=dropout_encoder
            )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=n_encoder_layers, 
            norm=None
            )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=dim_val,
            nhead=n_heads,
            dim_feedforward=dim_feedforward_decoder,
            dropout=dropout_decoder
            )

        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=n_decoder_layers, 
            norm=None
            )

    def forward(self, batch_x, src_mask, tgt_mask): 
        """
        Args:
            src: the encoder's output sequence.
            tgt: the sequence to the decoder.
            src_mask: the mask for the src sequence to prevent the model from 
                      using data points from the target sequence
            tgt_mask: the mask for the tgt sequence to prevent the model from
                      using data points from the target sequence
        """
        src = batch_x[:, :self.enc_seq_len, :]
        tgt = batch_x[:, -self.dec_seq_len: , :]


        # Pass throguh the input layer right before the encoder
        src = self.encoder_input_layer(src)

        # Pass through the positional encoding layer
        src = self.positional_encoding_layer(src)

        src = self.encoder(src=src)

        # Pass decoder input through decoder input layer
        decoder_output = self.decoder_input_layer(tgt)

        # Pass throguh decoder
        decoder_output = self.decoder(
            tgt=decoder_output,
            memory=src,
            tgt_mask=tgt_mask,
            memory_mask=src_mask
            )

        # Pass through the linear mapping layer
        decoder_output= self.linear_mapping(decoder_output)

        return decoder_output[:, -self.out_seq_len:, :]
    def generate_square_subsequent_mask(dim1: int, dim2: int) -> Tensor:
    
        return paddle.triu(paddle.ones((dim1, dim2)) * float('-inf'), diagonal=1)
