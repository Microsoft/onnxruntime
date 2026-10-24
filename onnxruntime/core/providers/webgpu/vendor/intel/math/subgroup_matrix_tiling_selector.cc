// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>

#include "core/providers/webgpu/compute_context.h"
#include "core/providers/webgpu/math/subgroup_matrix_tiling.h"
#include "core/providers/webgpu/vendor/intel/intel_device_info.h"
#include "core/providers/webgpu/vendor/intel/math/subgroup_matrix_tiling_selector.h"

// Pretuned tile + split-K table baked into the build; consulted before the
// heuristic in SelectTiling.
#include "core/providers/webgpu/vendor/intel/math/subgroup_matrix_matmul_tuned.inc"

namespace onnxruntime {
namespace webgpu {
namespace intel {

namespace {

// Subgroup matrix configuration used by this implementation (Intel Xe2/Xe3, F16).
constexpr uint32_t kSubgroupMatrixM = 8;
constexpr uint32_t kSubgroupMatrixN = 16;
constexpr uint32_t kSubgroupMatrixK = 16;

constexpr uint32_t kTileMCandidates[] = {8, 16, 32, 64};
constexpr uint32_t kTileNCandidates[] = {16, 32, 64};
constexpr uint32_t kSplitKCandidates[] = {1, 2, 4, 8};
// Per-tile-row a_scratch footprint, in elements, beyond the output scratch tile:
// 0 for MatMul (no im2col A-tile staging); kSubgroupMatrixK for Conv (width of
// its im2col A-tile staging buffer).
constexpr uint32_t kMatMulAScratchCols = 0;
constexpr uint32_t kConvAScratchCols = kSubgroupMatrixK;

// Workgroup-memory footprint of a tiling, in f16 elements.
uint64_t ScratchElems(const SubgroupMatrixTiling& t, uint32_t a_scratch_cols) {
  return static_cast<uint64_t>(t.split_k) * t.tile_m * (t.tile_n + a_scratch_cols);
}

bool IsTilingValid(const SubgroupMatrixTiling& t, uint32_t K, uint32_t a_scratch_cols) {
  if (t.tile_m == 0 || t.tile_n == 0 ||
      t.tile_m % kSubgroupMatrixM != 0 || t.tile_n % kSubgroupMatrixN != 0) {
    return false;
  }
  const uint32_t k_blocks = K / kSubgroupMatrixK;
  if (t.split_k == 0 || t.split_k > k_blocks) {
    return false;
  }
  return ScratchElems(t, a_scratch_cols) <= kMaxScratchElems;
}

// Falls back to a conservative default when HwSubgroups doesn't recognize arch.
uint32_t EffectiveHwSubgroups(std::string_view arch) {
  const uint32_t hw = HwSubgroups(arch);
  return hw == 0 ? 256u : hw;
}

// Fallback MatMul tiling: picks the tile whose independent output-tile grid just
// fills the resident subgroups, then adds split-K up to 2x occupancy. Not shared
// with ConvHeuristicTiling: shrinking tile_n for occupancy is free for MatMul but
// not Conv (see that function's comment).
SubgroupMatrixTiling HeuristicTiling(std::string_view arch, uint32_t M, uint32_t N, uint32_t K, uint32_t batch) {
  const uint32_t hw = EffectiveHwSubgroups(arch);
  const uint32_t k_blocks = K / kSubgroupMatrixK;
  auto tile_count = [](uint32_t dim, uint32_t tile) { return (dim + tile - 1) / tile; };

  // Skip tiles larger than the dimension. Prefer the largest-area tile that
  // already fills the hardware on its own; else the one with the most tiles.
  uint32_t tile_m = kTileMCandidates[0];
  uint32_t tile_n = kTileNCandidates[0];
  uint32_t best_tiles = 0;  // most output tiles seen so far (used only while under-filling)
  uint32_t best_area = 0;   // tile_m * tile_n of the current pick (tie-breaker: larger reuse wins)
  bool filled = false;      // true once some tile reaches num_tiles >= hw (machine is filled)
  for (uint32_t tm : kTileMCandidates) {
    if (tm > M && tm != kTileMCandidates[0]) {
      continue;
    }
    for (uint32_t tn : kTileNCandidates) {
      if (tn > N && tn != kTileNCandidates[0]) {
        continue;
      }
      const uint32_t tiles = batch * tile_count(M, tm) * tile_count(N, tn);
      const uint32_t area = tm * tn;
      if (tiles >= hw) {
        // This tile fills the machine on its own. Among all such tiles, keep the
        // largest-area one (best operand reuse); this also supersedes any
        // under-filling pick made earlier.
        if (!filled || area > best_area) {
          filled = true;
          best_area = area;
          tile_m = tm;
          tile_n = tn;
        }
      } else if (!filled && (tiles > best_tiles || (tiles == best_tiles && area > best_area))) {
        // No tile has filled the machine yet: chase occupancy by maximizing the
        // number of independent tiles, breaking ties toward larger area.
        best_tiles = tiles;
        best_area = area;
        tile_m = tm;
        tile_n = tn;
      }
    }
  }

  // Add split-K only while the independent tiles under-fill the machine; cap the
  // total at 2x hardware and keep enough K work per subgroup and scratch budget.
  const uint32_t num_tiles = batch * tile_count(M, tile_m) * tile_count(N, tile_n);
  constexpr uint32_t kMinBlocksPerSplit = 2;
  uint32_t split_k = 1;
  for (uint32_t cand : kSplitKCandidates) {
    if (k_blocks >= cand * kMinBlocksPerSplit &&
        num_tiles * cand <= 2 * hw &&
        ScratchElems({tile_m, tile_n, cand}, kMatMulAScratchCols) <= kMaxScratchElems) {
      split_k = cand;
    }
  }
  return {tile_m, tile_n, split_k};
}

// Conv's sole tiling strategy (a per-architecture tuned tree used to exist but
// didn't generalize). Unlike HeuristicTiling, never shrinks tile_n for occupancy:
// the im2col A tile is re-gathered per N tile, so gather volume (M*N*K/tile_n)
// would multiply -- tile_n is fixed at the largest candidate that fits N instead,
// tile_m at the smallest. split-K likewise has no occupancy cap (it hides gather
// latency independent of occupancy): just the largest candidate with enough work
// per split and scratch headroom.
SubgroupMatrixTiling ConvHeuristicTiling(uint32_t N, uint32_t K) {
  const uint32_t k_blocks = K / kSubgroupMatrixK;

  uint32_t tile_n = kTileNCandidates[0];
  for (uint32_t tn : kTileNCandidates) {
    if (tn <= N) {
      tile_n = tn;
    }
  }
  const uint32_t tile_m = kTileMCandidates[0];

  constexpr uint32_t kMinBlocksPerSplit = 2;
  uint32_t split_k = 1;
  for (uint32_t cand : kSplitKCandidates) {
    if (k_blocks >= cand * kMinBlocksPerSplit &&
        ScratchElems({tile_m, tile_n, cand}, kConvAScratchCols) <= kMaxScratchElems) {
      split_k = cand;
    }
  }
  return {tile_m, tile_n, split_k};
}

// Caps tile_n at the largest multiple of kSubgroupMatrixN that fits in N.
// Correctness, not performance: the Conv kernel's unchecked B load shifts a
// trailing partial N tile back to N - tile_n, which needs tile_n <= N, and N
// itself need not be a multiple of kSubgroupMatrixN. Callers must have already
// rejected N < kSubgroupMatrixN.
void ClampTileNToProblem(SubgroupMatrixTiling& t, uint32_t N) {
  const uint32_t max_tile_n = (N / kSubgroupMatrixN) * kSubgroupMatrixN;
  if (t.tile_n > max_tile_n) {
    t.tile_n = max_tile_n;
  }
}

// Looks up a pretuned tiling (subgroup_matrix_matmul_tuned.inc): match `arch`, map
// each dimension to the smallest grid value >= it. Nullopt if not covered.
std::optional<SubgroupMatrixTiling> LookupPretunedTiling(std::string_view arch, uint32_t M, uint32_t N, uint32_t K) {
  for (const auto& table : sgmm_tuned::kArchTables) {
    if (table.arch != arch) {
      continue;
    }
    auto bucket = [&table](uint32_t d) -> int {
      int last = 0;
      for (size_t i = 0; i < table.grid_count; ++i) {
        if (table.grid[i] >= d) {
          return static_cast<int>(i);
        }
        last = static_cast<int>(i);
      }
      return last;
    };
    const int mi = bucket(M);
    const int ni = bucket(N);
    const int ki = bucket(K);
    for (size_t i = 0; i < table.entry_count; ++i) {
      const auto& e = table.entries[i];
      if (e.mi == mi && e.ni == ni && e.ki == ki) {
        return SubgroupMatrixTiling{e.tile_m, e.tile_n, e.split_k};
      }
    }
    break;  // arch matched but no entry for this problem; fall through to heuristic
  }
  return std::nullopt;
}

// batch is a pure occupancy multiplier (one output-tile grid per z slice); retires
// split-K factors whose batch-scaled occupancy would exceed ~2x hardware.
// MatMul-only: Conv's tiling ignores occupancy and batch entirely.
void ClampSplitKForBatch(SubgroupMatrixTiling& t, uint32_t M, uint32_t N, uint32_t batch, uint32_t hw) {
  auto tile_count = [](uint32_t dim, uint32_t tile) { return (dim + tile - 1) / tile; };
  const uint32_t eff_tiles = batch * tile_count(M, t.tile_m) * tile_count(N, t.tile_n);
  while (t.split_k > 1 && eff_tiles * t.split_k > 2 * hw) {
    t.split_k /= 2;
  }
}

// Uses the pretuned table entry when one exists and fits, else the heuristic.
SubgroupMatrixTiling SelectTiling(std::string_view arch, uint32_t M, uint32_t N, uint32_t K, uint32_t batch) {
  if (const auto tuned = LookupPretunedTiling(arch, M, N, K); tuned && IsTilingValid(*tuned, K, kMatMulAScratchCols)) {
    SubgroupMatrixTiling tiling = *tuned;
    // The table is tuned at batch 1, so only revisit split-K when batch > 1.
    if (batch > 1) {
      ClampSplitKForBatch(tiling, M, N, batch, EffectiveHwSubgroups(arch));
    }
    return tiling;
  }
  return HeuristicTiling(arch, M, N, K, batch);
}

// Conv counterpart of SelectTiling: no pretuned table, always ConvHeuristicTiling.
SubgroupMatrixTiling SelectConvTiling(uint32_t N, uint32_t K) {
  SubgroupMatrixTiling tiling = ConvHeuristicTiling(N, K);
  ClampTileNToProblem(tiling, N);
  return tiling;
}

}  // namespace

SubgroupMatrixTilingSelector CreateSubgroupMatrixTilingSelector(
    const ComputeContextBase& context) {
  if (context.AdapterInfo().vendor != std::string_view{"intel"}) {
    return nullptr;
  }
  return [](const ComputeContext& context, uint32_t M, uint32_t N,
            uint32_t K, uint32_t batch) -> std::optional<SubgroupMatrixTiling> {
    // M/N partial tiles are handled by bounds-checked stores in the kernel.
    if (K % kSubgroupMatrixK != 0) {
      return std::nullopt;
    }
    const std::string_view arch = std::string_view{context.AdapterInfo().architecture};
    return SelectTiling(arch, M, N, K, batch);
  };
}

SubgroupMatrixTilingSelector CreateSubgroupMatrixConvTilingSelector(
    const ComputeContextBase& context) {
  if (context.AdapterInfo().vendor != std::string_view{"intel"}) {
    return nullptr;
  }
  return [](const ComputeContext& /*context*/, uint32_t /*M*/, uint32_t N,
            uint32_t K, uint32_t /*batch*/) -> std::optional<SubgroupMatrixTiling> {
    // N must hold at least one subgroup-matrix N block (ClampTileNToProblem can't
    // help below that); must agree with CanApplySubgroupMatrixConv's own check.
    if (K % kSubgroupMatrixK != 0 || N < kSubgroupMatrixN) {
      return std::nullopt;
    }
    return SelectConvTiling(N, K);
  };
}

}  // namespace intel
}  // namespace webgpu
}  // namespace onnxruntime
