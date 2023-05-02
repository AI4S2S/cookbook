"""Transformer with multi head attention."""


import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as f


def scaled_dot_product_attention(query: Tensor, key: Tensor, value: Tensor) -> Tensor:
    """
    Compute scaled dot-product attention between the query, key, and value tensors.

    Args:
        query (Tensor): The query tensor of shape (batch_size, num_query, query_dim).
        key (Tensor): The key tensor of shape (batch_size, num_key, key_dim).
        value (Tensor): The value tensor of shape (batch_size, num_value, value_dim).

    Returns:
        Tensor: The attention tensor of shape (batch_size, num_query, value_dim).
    """
    temp = query.bmm(key.transpose(1, 2))
    scale = query.size(-1) ** 0.5
    softmax = f.softmax(temp / scale, dim=-1)
    return softmax.bmm(value)


class AttentionHead(nn.Module):
    def __init__(self, dim_in: int, dim_q: int, dim_k: int):
        """
        A single attention head that computes attention scores between the query, key, and value tensors.

        Args:
            dim_in (int): The input dimension of the query, key, and value tensors.
            dim_q (int): The output dimension of the query tensor.
            dim_k (int): The output dimension of the key and value tensors.
        """
        super().__init__()
        self.q = nn.Linear(dim_in, dim_q)
        self.k = nn.Linear(dim_in, dim_k)
        self.v = nn.Linear(dim_in, dim_k)

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        """
        Compute attention scores between the query, key, and value tensors.

        Args:
            query (Tensor): The query tensor of shape (batch_size, num_query, query_dim).
            key (Tensor): The key tensor of shape (batch_size, num_key, key_dim).
            value (Tensor): The value tensor of shape (batch_size, num_value, value_dim).

        Returns:
            Tensor: The attention tensor of shape (batch_size, num_query, value_dim).
        """
        return scaled_dot_product_attention(self.q(query), self.k(key), self.v(value))


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, dim_in: int, dim_q: int, dim_k: int):
        """
        Initializes a Multi-Head Attention module.
        
        Args:
        - num_heads (int): The number of attention heads to use.
        - dim_in (int): The dimensionality of the input tensor.
        - dim_q (int): The dimensionality of the query tensor.
        - dim_k (int): The dimensionality of the key tensor.
        """
        super().__init__()
        self.heads = nn.ModuleList(
            [AttentionHead(dim_in, dim_q, dim_k) for _ in range(num_heads)]
        )
        self.linear = nn.Linear(num_heads * dim_k, dim_in)

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        """
        Computes the forward pass of the Multi-Head Attention module.
        
        Args:
        - query (Tensor): The query tensor of shape (batch_size, seq_len_q, dim_q).
        - key (Tensor): The key tensor of shape (batch_size, seq_len_k, dim_k).
        - value (Tensor): The value tensor of shape (batch_size, seq_len_v, dim_v).
        
        Returns:
        - output (Tensor): The output tensor of shape (batch_size, seq_len_q, dim_in).
        """
        return self.linear(
            torch.cat([h(query, key, value) for h in self.heads], dim=-1)
        )


def position_encoding(
    seq_len: int,
    dim_model: int,
    device: torch.device = torch.device("cpu"),
) -> Tensor:
    """
    Generate the positional encoding for a given sequence length and dimensionality.

    Args:
        seq_len (int): The length of the input sequence.
        dim_model (int): The dimensionality of the input sequence.
        device (torch.device, optional): The device on which to create the tensor.

    Returns:
        Tensor: The positional encoding tensor of shape (1, seq_len, dim_model).
    """
    pos = torch.arange(seq_len, dtype=torch.float, device=device).reshape(1, -1, 1)
    dim = torch.arange(dim_model, dtype=torch.float, device=device).reshape(1, 1, -1)
    phase = pos / (1e4 ** (dim / dim_model))

    return torch.where(dim.long() % 2 == 0, torch.sin(phase), torch.cos(phase))


def feed_forward(
    dim_input: int = 512, dim_feedforward: int = 2048, activation: nn.Module = nn.ReLU()
) -> nn.Module:
    """
    Create a feedforward network module with two linear layers and an activation function.

    Args:
        dim_input (int, optional): The dimensionality of the input tensor.
        dim_feedforward (int, optional): The dimensionality of the hidden layer.
        activation (nn.Module, optional): The activation function to use.

    Returns:
        nn.Module: The feedforward network module.
    """
    return nn.Sequential(
        nn.Linear(dim_input, dim_feedforward),
        activation,
        nn.Linear(dim_feedforward, dim_input),
    )


class Residual(nn.Module):
    def __init__(self, sublayer: nn.Module, dimension: int, dropout: float = 0.1):
        """
        Apply residual connections to a sublayer module.

        Args:
            sublayer (nn.Module): The sublayer module to apply the residual connection to.
            dimension (int): The dimensionality of the sublayer input/output tensors.
            dropout (float, optional): The dropout probability to apply (default: 0.1).
        """
        super().__init__()
        self.sublayer = sublayer
        self.norm = nn.LayerNorm(dimension)
        self.dropout = nn.Dropout(dropout)

    def forward(self, *tensors: Tensor) -> Tensor:
        """
        Compute a forward pass through the residual module.

        Args:
            tensors (Tuple[Tensor]): The input tensor(s) to the sublayer.

        Returns:
            Tensor: The output tensor of the residual module.
        """
        # Assume that the "query" tensor is given first, so we can compute the
        # residual.  This matches the signature of 'MultiHeadAttention'.
        return self.norm(tensors[0] + self.dropout(self.sublayer(*tensors)))


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        dim_model: int = 512,
        num_heads: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: nn.Module = nn.ReLU(),
    ):
        super().__init__()
        dim_q = dim_k = max(dim_model // num_heads, 1)
        self.attention = Residual(
            MultiHeadAttention(num_heads, dim_model, dim_q, dim_k),
            dimension=dim_model,
            dropout=dropout,
        )
        self.feed_forward = Residual(
            feed_forward(dim_model, dim_feedforward, activation),
            dimension=dim_model,
            dropout=dropout,
        )

    def forward(self, src: Tensor) -> Tensor:
        src = self.attention(src, src, src)
        return self.feed_forward(src)


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        num_layers: int = 6,
        dim_model: int = 512,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: nn.Module = nn.ReLU(),
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    dim_model, num_heads, dim_feedforward, dropout, activation
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, src: Tensor) -> Tensor:
        seq_len, dimension = src.size(1), src.size(2)
        src += position_encoding(seq_len, dimension)
        for layer in self.layers:
            src = layer(src)

        return src


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        dim_model: int = 512,
        num_heads: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: nn.Module = nn.ReLU(),
    ):
        super().__init__()
        dim_q = dim_k = max(dim_model // num_heads, 1)
        self.attention_1 = Residual(
            MultiHeadAttention(num_heads, dim_model, dim_q, dim_k),
            dimension=dim_model,
            dropout=dropout,
        )
        self.attention_2 = Residual(
            MultiHeadAttention(num_heads, dim_model, dim_q, dim_k),
            dimension=dim_model,
            dropout=dropout,
        )
        self.feed_forward = Residual(
            feed_forward(dim_model, dim_feedforward, activation),
            dimension=dim_model,
            dropout=dropout,
        )

    def forward(self, tgt: Tensor, memory: Tensor) -> Tensor:
        tgt = self.attention_1(tgt, tgt, tgt)
        tgt = self.attention_2(tgt, memory, memory)
        return self.feed_forward(tgt)


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        num_layers: int = 6,
        dim_model: int = 512,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: nn.Module = nn.ReLU(),
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerDecoderLayer(
                    dim_model, num_heads, dim_feedforward, dropout, activation
                )
                for _ in range(num_layers)
            ]
        )
        self.linear = nn.Linear(dim_model, dim_model)

    def forward(self, tgt: Tensor, memory: Tensor) -> Tensor:
        seq_len, dimension = tgt.size(1), tgt.size(2)
        tgt += position_encoding(seq_len, dimension)
        for layer in self.layers:
            tgt = layer(tgt, memory)

        return torch.softmax(self.linear(tgt), dim=-1)


class Transformer(nn.Module):
    def __init__(
        self,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_model: int = 512,
        num_heads: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: nn.Module = nn.ReLU(),
    ):
        super().__init__()
        self.encoder = TransformerEncoder(
            num_layers=num_encoder_layers,
            dim_model=dim_model,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
        )
        self.decoder = TransformerDecoder(
            num_layers=num_decoder_layers,
            dim_model=dim_model,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
        )

    def forward(self, src: Tensor, tgt: Tensor) -> Tensor:
        return self.decoder(tgt, self.encoder(src))
