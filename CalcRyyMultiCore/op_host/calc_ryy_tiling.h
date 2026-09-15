#ifndef CALC_RYY_TILING_H
#define CALC_RYY_TILING_H

#include <cstdint>

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace CalcRyyConst {
constexpr int32_t ROWS = 8;
constexpr int32_t COLS = 32;
constexpr int32_t LINKS_PER_CHUNK = 8;
constexpr int32_t PACK_M = ROWS * 2;
constexpr int32_t PACK_N = 16;
constexpr int32_t PACK_K = COLS * 2 * LINKS_PER_CHUNK;
constexpr int32_t GROUPS_PER_CHUNK = 16;
constexpr int32_t DEFAULT_BLOCK_DIM = GROUPS_PER_CHUNK;

constexpr uint64_t A_BLOCK_ELEMS = static_cast<uint64_t>(PACK_M) * PACK_K;
constexpr uint64_t B_BLOCK_ELEMS = static_cast<uint64_t>(PACK_N) * PACK_K;
constexpr uint64_t C_BLOCK_ELEMS = static_cast<uint64_t>(PACK_M) * PACK_N;
constexpr uint64_t ACC_BLOCK_ELEMS = static_cast<uint64_t>(PACK_M) * ROWS;
constexpr uint64_t WORKSPACE_ELEMS_PER_GROUP =
    A_BLOCK_ELEMS + B_BLOCK_ELEMS + C_BLOCK_ELEMS + ACC_BLOCK_ELEMS;
constexpr uint64_t WORKSPACE_BYTES_PER_GROUP = WORKSPACE_ELEMS_PER_GROUP * sizeof(float);
constexpr uint64_t WORKSPACE_BYTES = GROUPS_PER_CHUNK * WORKSPACE_BYTES_PER_GROUP;
} // namespace CalcRyyConst

namespace optiling {
BEGIN_TILING_DATA_DEF(CalcRyyTilingData)
TILING_DATA_FIELD_DEF(uint32_t, aNum);
TILING_DATA_FIELD_DEF(uint32_t, bNum);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, cubeTilingData);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(CalcRyy, CalcRyyTilingData)
} // namespace optiling

#endif // CALC_RYY_TILING_H
