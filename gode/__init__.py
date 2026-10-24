"""GoDe: backbone-agnostic progressive compression of radiance-field primitives.

The package is independent of the primitive representation. A backbone has to
expose: render(camera, mask=...), training_setup/update_learning_rate/optimizer,
train()/eval(), num_primitives/primitive_names()/index_primitives()/subset()/
append_primitives()/clone_empty(), _get_gradient_scores()/gradient_prune(), _sort_morton(),
set_qa()/quantizer_names()/calibrate_quantizers_for_compression(),
appearance_regularization(), and encode_payload()/decode_payload().
"""

from gode.compression import (decode_all_levels, decode_and_evaluate,
                              decode_levels, encode, snapshot)
from gode.evaluation import baseline_row, evaluate, report, write_csv
from gode.hierarchy import build as build_hierarchy
from gode.trainer import train as train_scalable, warmup as qat_warmup

__all__ = [
    "build_hierarchy", "train_scalable", "qat_warmup", "snapshot", "encode",
    "decode_and_evaluate", "decode_levels", "decode_all_levels", "evaluate",
    "baseline_row",
    "report", "write_csv",
]
