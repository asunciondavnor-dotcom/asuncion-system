"""CPU inference for the checksum-pinned MiniLM ONNX export."""

import hashlib
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer


MAPPED_ORT_VERSION = '1.32.0'


def cpu_options():
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.add_session_config_entry('session.intra_op.allow_spinning','0')
    return options


class MiniLMEncoder:
    def __init__(self, model_root, artifact, expected_sha256):
        # Release artifacts are read-only and checked before graph loading.
        with artifact.open('rb') as serialized:
            if hashlib.file_digest(serialized,'sha256').hexdigest()!=expected_sha256:
                raise ValueError('PR_PO_ONNX_ARTIFACT_INTEGRITY')
        options = cpu_options()
        if artifact.suffix=='.ort':
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            options.add_session_config_entry('session.load_model_format','ORT')
            # Skip duplicating/packing constant weights on a throttled cold start.
            options.add_session_config_entry('session.disable_prepacking','1')
            if ort.__version__ == MAPPED_ORT_VERSION:
                # The verified graph remains read-only for the life of this session.
                options.add_session_config_entry('session.use_memory_mapped_ort_model','1')
                options.add_session_config_entry('session.use_ort_model_bytes_for_initializers','1')
        else:
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        self.tokenizer = Tokenizer.from_file(str(model_root/'tokenizer.json'))
        self.tokenizer.enable_truncation(max_length=256,strategy='longest_first')
        self.tokenizer.enable_padding(pad_id=0,pad_token='[PAD]')
        self.session = ort.InferenceSession(str(artifact),sess_options=options,providers=['CPUExecutionProvider'])
        if [entry.name for entry in self.session.get_inputs()]!=['ids','types','mask']:
            raise ValueError('PR_PO_ONNX_INPUT_CONTRACT')

    def encode(self, sentences, normalize_embeddings=False):
        order = np.argsort([-len(sentence) for sentence in sentences])
        embeddings = np.empty((len(order),384),dtype=np.float32)
        for start in range(0,len(order),32):
            positions = order[start:start+32]
            rows = self.tokenizer.encode_batch([sentences[i] for i in positions])
            batch_size = max(1,512//len(rows[0].ids))
            for first in range(0,len(rows),batch_size):
                chunk = slice(first,first+batch_size)
                group = rows[chunk]
                inputs = {'ids':np.asarray([row.ids for row in group],dtype=np.int64),
                          'types':np.asarray([row.type_ids for row in group],dtype=np.int64),
                          'mask':np.asarray([row.attention_mask for row in group],dtype=np.int64)}
                embeddings[positions[chunk]] = self.session.run(['embeddings'],inputs)[0]
        if normalize_embeddings:
            embeddings /= np.maximum(np.linalg.norm(embeddings,axis=1,keepdims=True),np.float32(1e-12))
        return embeddings
