"""
model.py - Transformer Architecture Skeleton
DA6401 Assignment 3: "Attention Is All You Need"

AUTOGRADER CONTRACT (DO NOT MODIFY SIGNATURES):
  scaled_dot_product_attention(Q, K, V, mask) -> (out, weights)
  MultiHeadAttention.forward(q, k, v, mask)   -> Tensor
  PositionalEncoding.forward(x)               -> Tensor
  make_src_mask(src, pad_idx)                 -> BoolTensor
  make_tgt_mask(tgt, pad_idx)                 -> BoolTensor
  Transformer.encode(src, src_mask)           -> Tensor
  Transformer.decode(memory,src_m,tgt,tgt_m)  -> Tensor
"""

import copy
import math
import re
from typing import Optional, Tuple

import torch
import torch.nn as nn


class _InferenceVocab:
    def __init__(self, itos: list[str], stoi: dict[str, int]) -> None:
        self.itos = itos
        self.stoi = stoi
        self.idx_to_token = itos
        self.token_to_idx = stoi
        self.unk_idx = stoi["<unk>"]
        self.pad_idx = stoi["<pad>"]
        self.sos_idx = stoi["<sos>"]
        self.eos_idx = stoi["<eos>"]

    def lookup_token(self, idx: int) -> str:
        if 0 <= idx < len(self.itos):
            return self.itos[idx]
        return "<unk>"

    def lookup_indices(self, tokens: list[str]) -> list[int]:
        return [self.stoi.get(token, self.unk_idx) for token in tokens]


def _fallback_tokenize(text: str) -> list[str]:
    return re.findall(r"\w+|[^\w\s]", text.lower(), flags=re.UNICODE)


def _build_inference_assets():
    from fast_vocab import SRC_ITOS, SRC_STOI, TGT_ITOS, TGT_STOI

    return {
        "src_vocab": _InferenceVocab(SRC_ITOS, SRC_STOI),
        "tgt_vocab": _InferenceVocab(TGT_ITOS, TGT_STOI),
        "src_tokenizer": _fallback_tokenize,
    }


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    use_scale: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Scaled Dot-Product Attention.

        Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V

    Args:
        Q    : Query tensor,  shape (..., seq_q, d_k)
        K    : Key tensor,    shape (..., seq_k, d_k)
        V    : Value tensor,  shape (..., seq_k, d_v)
        mask : Optional Boolean mask, shape broadcastable to
               (..., seq_q, seq_k).
               Positions where mask is True are masked out.

    Returns:
        output : Attended output,   shape (..., seq_q, d_v)
        attn_w : Attention weights, shape (..., seq_q, seq_k)
    """
    d_k = Q.size(-1)
    scale = math.sqrt(d_k) if use_scale else 1.0
    scores = torch.matmul(Q, K.transpose(-2, -1)) / scale

    if mask is not None:
        mask = mask.to(device=scores.device, dtype=torch.bool)
        scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)

    attn_weights = torch.softmax(scores, dim=-1)

    if mask is not None:
        attn_weights = attn_weights.masked_fill(mask, 0.0)
        attn_weights = attn_weights / attn_weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)

    output = torch.matmul(attn_weights, V)
    return output, attn_weights


def make_src_mask(
    src: torch.Tensor,
    pad_idx: int = 1,
) -> torch.Tensor:
    """
    Build a padding mask for the encoder (source sequence).

    Returns:
        Boolean mask, shape [batch, 1, 1, src_len]
        True  -> position is a PAD token (will be masked out)
        False -> real token
    """
    return (src == pad_idx).unsqueeze(1).unsqueeze(2)


def make_tgt_mask(
    tgt: torch.Tensor,
    pad_idx: int = 1,
) -> torch.Tensor:
    """
    Build a combined padding + causal (look-ahead) mask for the decoder.

    Returns:
        Boolean mask, shape [batch, 1, tgt_len, tgt_len]
        True -> position is masked out (PAD or future token)
    """
    batch_size, tgt_len = tgt.shape
    pad_mask = (tgt == pad_idx).unsqueeze(1).unsqueeze(2)
    causal_mask = torch.triu(
        torch.ones(tgt_len, tgt_len, device=tgt.device, dtype=torch.bool),
        diagonal=1,
    ).unsqueeze(0).unsqueeze(0)
    return pad_mask.expand(batch_size, 1, tgt_len, tgt_len) | causal_mask


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention as in "Attention Is All You Need".

    Args:
        d_model   (int)  : Total model dimensionality. Must be divisible by num_heads.
        num_heads (int)  : Number of parallel attention heads h.
        dropout   (float): Dropout probability applied to attention weights.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1, use_scale: bool = True) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.use_scale = use_scale

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.attn_weights: Optional[torch.Tensor] = None

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        return x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

    def _combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, seq_len, _ = x.shape
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query : shape [batch, seq_q, d_model]
            key   : shape [batch, seq_k, d_model]
            value : shape [batch, seq_k, d_model]
            mask  : Optional BoolTensor broadcastable to
                    [batch, num_heads, seq_q, seq_k]
                    True -> masked out

        Returns:
            output : shape [batch, seq_q, d_model]
        """
        q = self._split_heads(self.W_q(query))
        k = self._split_heads(self.W_k(key))
        v = self._split_heads(self.W_v(value))

        _, attn_weights = scaled_dot_product_attention(q, k, v, mask, use_scale=self.use_scale)
        self.attn_weights = attn_weights

        attn_output = torch.matmul(self.dropout(attn_weights), v)
        attn_output = self._combine_heads(attn_output)
        return self.W_o(attn_output)


class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding as in "Attention Is All You Need".

    Args:
        d_model  (int)  : Embedding dimensionality.
        dropout  (float): Dropout applied after adding encodings.
        max_len  (int)  : Maximum sequence length to pre-compute.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : Input embeddings, shape [batch, seq_len, d_model]

        Returns:
            Tensor of same shape [batch, seq_len, d_model]
        """
        seq_len = x.size(1)
        if seq_len > self.pe.size(1):
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_len {self.pe.size(1)} "
                "configured for positional encoding."
            )
        x = x + self.pe[:, :seq_len].to(dtype=x.dtype)
        return self.dropout(x)


class LearnedPositionalEncoding(nn.Module):
    """
    Learned Positional Encoding (Task 2.4 Ablation).
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = x + self.pe(positions)
        return self.dropout(x)


class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network:

        FFN(x) = max(0, x W1 + b1) W2 + b2
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class EncoderLayer(nn.Module):
    """
    Single Transformer encoder layer.

    This implementation uses Pre-LayerNorm for improved training stability.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1, use_scale: bool = True) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout, use_scale=use_scale)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        norm_x = self.norm1(x)
        x = x + self.dropout1(self.self_attn(norm_x, norm_x, norm_x, src_mask))
        x = x + self.dropout2(self.feed_forward(self.norm2(x)))
        return x


class DecoderLayer(nn.Module):
    """
    Single Transformer decoder layer.

    This implementation uses Pre-LayerNorm for improved training stability.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1, use_scale: bool = True) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout, use_scale=use_scale)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout, use_scale=use_scale)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        norm_x = self.norm1(x)
        x = x + self.dropout1(self.self_attn(norm_x, norm_x, norm_x, tgt_mask))

        norm_x = self.norm2(x)
        x = x + self.dropout2(self.cross_attn(norm_x, memory, memory, src_mask))

        x = x + self.dropout3(self.feed_forward(self.norm3(x)))
        return x


class Encoder(nn.Module):
    """Stack of N identical EncoderLayer modules with final LayerNorm."""

    def __init__(self, layer: EncoderLayer, N: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = nn.LayerNorm(layer.self_attn.d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    """Stack of N identical DecoderLayer modules with final LayerNorm."""

    def __init__(self, layer: DecoderLayer, N: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = nn.LayerNorm(layer.self_attn.d_model)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class Transformer(nn.Module):
    """
    Full Encoder-Decoder Transformer for sequence-to-sequence tasks.
    """

    _cached_inference_assets = None

    def __init__(
        self,
        src_vocab_size: int = None,
        tgt_vocab_size: int = None,
        d_model: int = 512,
        N: int = 6,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        pos_encoding_type: str = "sinusoidal",
        use_scale: bool = True,
    ) -> None:
        super().__init__()
        
        # Autograder compatibility: if not provided, try to load from checkpoint
        if src_vocab_size is None or tgt_vocab_size is None:
            import os
            import torch
            if os.path.exists("checkpoint.pt"):
                try:
                    checkpoint = torch.load("checkpoint.pt", map_location="cpu", weights_only=False)
                    config = checkpoint.get("model_config", {})
                    src_vocab_size = config.get("src_vocab_size", 7851)
                    tgt_vocab_size = config.get("tgt_vocab_size", 5892)
                    d_model = config.get("d_model", d_model)
                    N = config.get("N", N)
                    num_heads = config.get("num_heads", num_heads)
                    d_ff = config.get("d_ff", d_ff)
                    dropout = config.get("dropout", dropout)
                    pos_encoding_type = config.get("pos_encoding_type", pos_encoding_type)
                    use_scale = config.get("use_scale", use_scale)
                except Exception:
                    src_vocab_size = 7851
                    tgt_vocab_size = 5892
            else:
                src_vocab_size = 7851
                tgt_vocab_size = 5892

        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.d_model = d_model
        self.N = N
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout = dropout

        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        
        if pos_encoding_type == "learned":
            self.positional_encoding = LearnedPositionalEncoding(d_model, dropout)
        else:
            self.positional_encoding = PositionalEncoding(d_model, dropout)

        encoder_layer = EncoderLayer(d_model, num_heads, d_ff, dropout, use_scale=use_scale)
        decoder_layer = DecoderLayer(d_model, num_heads, d_ff, dropout, use_scale=use_scale)
        self.encoder = Encoder(encoder_layer, N)
        self.decoder = Decoder(decoder_layer, N)
        self.generator = nn.Linear(d_model, tgt_vocab_size)
        self.embedding_scale = math.sqrt(d_model)

        self.model_config = {
            "src_vocab_size": src_vocab_size,
            "tgt_vocab_size": tgt_vocab_size,
            "d_model": d_model,
            "N": N,
            "num_heads": num_heads,
            "d_ff": d_ff,
            "dropout": dropout,
            "pos_encoding_type": pos_encoding_type,
            "use_scale": use_scale,
        }

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)

    def encode(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor,
    ) -> torch.Tensor:
        src_embeddings = self.src_embedding(src) * self.embedding_scale
        src_embeddings = self.positional_encoding(src_embeddings)
        return self.encoder(src_embeddings, src_mask)

    def decode(
        self,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        tgt_embeddings = self.tgt_embedding(tgt) * self.embedding_scale
        tgt_embeddings = self.positional_encoding(tgt_embeddings)
        decoder_output = self.decoder(tgt_embeddings, memory, src_mask, tgt_mask)
        return self.generator(decoder_output)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        memory = self.encode(src, src_mask)
        return self.decode(memory, src_mask, tgt, tgt_mask)

    def infer(self, src_text: str, max_len: int = 100) -> str:
        """
        Greedy decoding inference for the autograder.
        """
        device = next(self.parameters()).device

        if Transformer._cached_inference_assets is None:
            Transformer._cached_inference_assets = _build_inference_assets()

        assets = Transformer._cached_inference_assets
        src_vocab = assets["src_vocab"]
        tgt_vocab = assets["tgt_vocab"]
        src_tokenizer = assets["src_tokenizer"]

        # Tokenize source text
        tokens = src_tokenizer(src_text)
        src_ids = [src_vocab.sos_idx] + src_vocab.lookup_indices(tokens) + [src_vocab.eos_idx]
        src_tensor = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0).to(device)

        # Create masks
        src_mask = make_src_mask(src_tensor, pad_idx=src_vocab.pad_idx)

        # Generate target tokens
        ys = torch.full((1, 1), tgt_vocab.sos_idx, dtype=torch.long, device=device)

        was_training = self.training
        self.eval()

        with torch.no_grad():
            memory = self.encode(src_tensor, src_mask)
            for _ in range(max_len - 1):
                tgt_mask = make_tgt_mask(ys, pad_idx=tgt_vocab.pad_idx)
                logits = self.decode(memory, src_mask, ys, tgt_mask)
                next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
                ys = torch.cat([ys, next_token], dim=1)
                if next_token.item() == tgt_vocab.eos_idx:
                    break

        self.train(was_training)

        # Detokenize target tokens
        tgt_indices = ys.squeeze(0).tolist()
        pred_tokens = []
        for idx in tgt_indices:
            if idx == tgt_vocab.eos_idx:
                break
            if idx not in {tgt_vocab.sos_idx, tgt_vocab.pad_idx}:
                pred_tokens.append(tgt_vocab.lookup_token(idx))

        return " ".join(pred_tokens)
