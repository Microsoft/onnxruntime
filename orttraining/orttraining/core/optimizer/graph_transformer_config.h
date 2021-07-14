
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/graph_transformer_config.h"

namespace onnxruntime {

namespace training {

struct TrainingGraphTransformerConfiguration : public GraphTransformerConfiguration {
  bool enable_gelu_approximation{false};

  // Enable recompute of attention dropout to save memory
  bool attn_dropout_recompute{false};
  // Enable recompute of Gelu activation output to save memory
  bool gelu_recompute{false};
  // Enable recompute of Expand to save memory
  bool expand_recompute{true};
  // Enable recompute of transformer layer ouput to save memory
  bool transformer_layer_recompute{false};
  // Number of layers to apply recompute
  int number_recompute_layers{0};
  bool allow_layer_norm_mod_precision{false};
};

}  // namespace training
}  // namespace onnxruntime
