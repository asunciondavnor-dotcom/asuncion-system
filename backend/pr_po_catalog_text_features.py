"""Hash short character patterns for the trained product identity model."""

from functools import lru_cache
from hashlib import blake2b
import re

import numpy as np


CHAR_WIDTH = 1536
WORD = re.compile(r'[^\W\d_]+')


@lru_cache(maxsize=2048)
def index(fragment):
    return int.from_bytes(blake2b(fragment.encode(),digest_size=4).digest(),'little') % CHAR_WIDTH


def characters(name):
    tokens = WORD.findall(name.casefold())
    value = np.zeros(CHAR_WIDTH,dtype=np.float32)
    for word in tokens:
        padded = f'^{word}$'
        for width in (2,3,4):
            for i in range(len(padded)-width+1):
                value[index(padded[i:i+width])] += 1
    strength = np.linalg.norm(value)
    return value / strength if strength else value


def product_phrase(name):
    """Keep the named article distinct from equipment mentioned after 'for'."""
    leading, delimiter, trailing = name.partition(' for ')
    return leading if delimiter and leading and trailing else name
