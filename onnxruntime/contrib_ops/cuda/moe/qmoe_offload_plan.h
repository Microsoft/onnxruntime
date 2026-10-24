#pragma once

#include <algorithm>
#include <cstddef>

namespace onnxruntime::contrib::cuda::qmoe {

template <typename T>
inline void TransposeOffloadScales(const T* source, T* destination, int columns, int blocks) {
  constexpr int column_tile = 32;
  constexpr int block_tile = 8;
  for (int column_start = 0; column_start < columns; column_start += column_tile) {
    const int column_end = std::min(columns, column_start + column_tile);
    for (int block_start = 0; block_start < blocks; block_start += block_tile) {
      const int block_end = std::min(blocks, block_start + block_tile);
      for (int column = column_start; column < column_end; ++column) {
        const auto* source_row = source + static_cast<size_t>(column) * blocks;
        for (int block = block_start; block < block_end; ++block) {
          destination[static_cast<size_t>(block) * columns + column] = source_row[block];
        }
      }
    }
  }
}

struct OffloadWave {
  int next_expert = 0;
  int expert_count = 0;
  int route_count = 0;
};

struct StreamExpertLocation {
  int buffer = -1;
  int slot = -1;
};

inline StreamExpertLocation FindLiveStreamExpert(int expert, const int* streamed_experts,
                                                 int streamed_count, int wave_size) {
  if (!streamed_experts || streamed_count <= 0 || wave_size <= 0) return {};
  const int waves = (streamed_count - 1) / wave_size + 1;
  // Only the final wave in each half of the two-buffer landing ring survives.
  const int first_live = std::max(0, waves - 2) * wave_size;
  const auto* end = streamed_experts + streamed_count;
  const auto* found = std::lower_bound(streamed_experts + first_live, end, expert);
  if (found == end || *found != expert) return {};
  const int index = static_cast<int>(found - streamed_experts);
  return {(index / wave_size) & 1, index % wave_size};
}

inline bool InitializeOffloadRoutes(const int* selected_experts, int route_count,
                                    int num_experts, int k, int* expert_to_slot) {
  if (route_count <= 0 || num_experts <= 0 || k <= 0 || k > num_experts || route_count % k != 0) {
    return false;
  }
  std::fill_n(expert_to_slot, num_experts, -1);
  for (int route = 0; route < route_count; ++route) {
    const int expert = selected_experts[route];
    if (expert < 0 || expert >= num_experts) {
      return false;
    }
    const int token_start = route - route % k;
    for (int previous = token_start; previous < route; ++previous) {
      if (selected_experts[previous] == expert) {
        return false;
      }
    }
    expert_to_slot[expert] = 0;
  }
  return true;
}

inline OffloadWave MakeOffloadWave(const int* selected_experts, int route_count,
                                   int num_experts, int first_expert, int wave_capacity,
                                   int* expert_to_slot, int* wave_experts,
                                   int* route_ids, int* compact_experts) {
  OffloadWave wave;
  wave.next_expert = first_expert;
  while (wave.next_expert < num_experts && wave.expert_count < wave_capacity) {
    const int expert = wave.next_expert++;
    if (expert_to_slot[expert] >= 0) {
      expert_to_slot[expert] = wave.expert_count;
      wave_experts[wave.expert_count++] = expert;
    }
  }
  for (int route = 0; route < route_count; ++route) {
    const int expert = selected_experts[route];
    if (expert >= first_expert && expert < wave.next_expert && expert_to_slot[expert] >= 0) {
      route_ids[wave.route_count] = route;
      compact_experts[wave.route_count++] = expert_to_slot[expert];
    }
  }
  return wave;
}

}  // namespace onnxruntime::contrib::cuda::qmoe
