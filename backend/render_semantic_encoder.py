"""CPU inference for the exact, verified MiniLM-L6-v2 Render bundle."""

import numpy as np
import torch
from safetensors.torch import load_file
from tokenizers import Tokenizer
from torch.nn import functional as F


HIDDEN_SIZE = 384
ATTENTION_HEADS = 12


class MiniLMEncoder:
    """Use the pinned tokenizer and transformer tensors without training imports."""

    def __init__(self, root):
        self.tokenizer = Tokenizer.from_file(str(root / 'tokenizer.json'))
        self.tokenizer.enable_truncation(max_length=256, strategy='longest_first')
        self.tokenizer.enable_padding(pad_id=0, pad_token='[PAD]')
        self.weights = load_file(str(root / 'model.safetensors'), device='cpu')

    def _linear(self, values, key):
        return F.linear(values, self.weights[key + '.weight'], self.weights[key + '.bias'])

    def _layer_norm(self, values, key):
        return F.layer_norm(values, (HIDDEN_SIZE,), self.weights[key + '.weight'],
                            self.weights[key + '.bias'], eps=1e-12)

    def _encode_tokens(self, ids, types, mask):
        batch, length = ids.shape
        positions = torch.arange(length).unsqueeze(0)
        hidden = F.embedding(ids, self.weights['embeddings.word_embeddings.weight'])
        hidden += F.embedding(types, self.weights['embeddings.token_type_embeddings.weight'])
        hidden += F.embedding(positions, self.weights['embeddings.position_embeddings.weight'])
        hidden = self._layer_norm(hidden, 'embeddings.LayerNorm')
        attention_mask = None if mask.all().item() else mask[:, None, None, :].bool()
        for index in range(6):
            prefix = f'encoder.layer.{index}'
            query, key, value = (
                self._linear(hidden, f'{prefix}.attention.self.{name}')
                .view(batch, length, ATTENTION_HEADS, HIDDEN_SIZE // ATTENTION_HEADS).transpose(1, 2)
                for name in ('query', 'key', 'value')
            )
            attention = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask)
            del query, key, value
            attention = attention.transpose(1, 2).contiguous().reshape(batch, length, HIDDEN_SIZE)
            hidden = self._layer_norm(self._linear(attention, prefix + '.attention.output.dense') + hidden,
                                      prefix + '.attention.output.LayerNorm')
            del attention
            intermediate = F.gelu(self._linear(hidden, prefix + '.intermediate.dense'))
            hidden = self._layer_norm(self._linear(intermediate, prefix + '.output.dense') + hidden,
                                      prefix + '.output.LayerNorm')
            del intermediate
        return hidden

    @torch.inference_mode()
    def encode(self, sentences, normalize_embeddings=False):
        if not len(sentences):
            return np.empty((0, HIDDEN_SIZE), dtype=np.float32)
        # Match the approved reference's sort direction and batch boundaries.
        order = np.argsort([-len(sentence) for sentence in sentences])
        outputs = []
        for start in range(0, len(order), 32):
            batch = self.tokenizer.encode_batch([sentences[i] for i in order[start:start + 32]])
            ids = torch.tensor([row.ids for row in batch], dtype=torch.long)
            types = torch.tensor([row.type_ids for row in batch], dtype=torch.long)
            mask = torch.tensor([row.attention_mask for row in batch], dtype=torch.long)
            tokens = self._encode_tokens(ids, types, mask)
            weights = mask.unsqueeze(-1).expand_as(tokens).to(tokens.dtype)
            vectors = (tokens * weights).sum(dim=1) / torch.clamp(weights.sum(dim=1), min=1e-9)
            vectors = F.normalize(vectors, p=2, dim=1)
            if normalize_embeddings:
                vectors = F.normalize(vectors, p=2, dim=1)
            outputs.append(vectors.numpy())
        return np.concatenate(outputs)[np.argsort(order)]
