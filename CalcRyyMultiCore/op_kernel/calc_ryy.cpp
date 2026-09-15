/*
 * CalcRyy
 *
 * h = h_r + i * h_i, h shape [aNum, 8, 32].  For every B item:
 *   ryy = sum_j(interf_power[offset + j] * h_j * h_j^H)
 *   diag(ryy) += weak_interf
 *   hs[:, :hsLen] = h[offset, :, hsoffset:hsoffset + hsLen]
 *
 * Eight links are packed into one real GEMM:
 *
 *   A = [p*Hr,  p*Hi]   B = [Hr]
 *       [p*Hi, -p*Hr]       [Hi]
 *
 *   A * B^T = [real(h*h^H); imag(h*h^H)].
 */
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/matmul.h"

using namespace AscendC;

namespace CalcRyyKernel {
constexpr uint32_t ROWS = 8;
constexpr uint32_t COLS = 32;
constexpr uint32_t LINKS_PER_CHUNK = 8;
constexpr uint32_t PACK_M = ROWS * 2;
constexpr uint32_t PACK_N = 16;
constexpr uint32_t PACK_K = COLS * 2 * LINKS_PER_CHUNK;
constexpr uint32_t GROUPS_PER_CHUNK = 16;

constexpr uint64_t H_MATRIX_ELEMS = static_cast<uint64_t>(ROWS) * COLS;
constexpr uint64_t OUTPUT_MATRIX_ELEMS = static_cast<uint64_t>(ROWS) * ROWS;
constexpr uint64_t A_BLOCK_ELEMS = static_cast<uint64_t>(PACK_M) * PACK_K;
constexpr uint64_t B_BLOCK_ELEMS = static_cast<uint64_t>(PACK_N) * PACK_K;
constexpr uint64_t C_BLOCK_ELEMS = static_cast<uint64_t>(PACK_M) * PACK_N;
constexpr uint64_t ACC_BLOCK_ELEMS = static_cast<uint64_t>(PACK_M) * ROWS;
constexpr uint64_t A_BLOCK_BYTES = A_BLOCK_ELEMS * sizeof(float);
constexpr uint64_t B_BLOCK_BYTES = B_BLOCK_ELEMS * sizeof(float);
constexpr uint64_t C_BLOCK_BYTES = C_BLOCK_ELEMS * sizeof(float);
constexpr uint64_t ACC_BLOCK_BYTES = ACC_BLOCK_ELEMS * sizeof(float);
constexpr uint64_t WORKSPACE_BYTES_PER_GROUP =
    A_BLOCK_BYTES + B_BLOCK_BYTES + C_BLOCK_BYTES + ACC_BLOCK_BYTES;

struct BInfo {
    uint32_t offset;
    uint32_t len;
    uint32_t hsOffset;
    uint32_t hsLen;
};

__aicore__ inline uint32_t MinU32(uint32_t lhs, uint32_t rhs)
{
    return lhs < rhs ? lhs : rhs;
}

__aicore__ inline uint32_t CeilDiv(uint32_t value, uint32_t divisor)
{
    return (value + divisor - 1) / divisor;
}

__aicore__ inline BInfo LoadBInfo(__gm__ uint32_t *infoGm, uint32_t bIdx)
{
    const uint64_t base = 3 + static_cast<uint64_t>(bIdx) * 4;
    BInfo info;
    info.offset = infoGm[base];
    info.len = infoGm[base + 1];
    info.hsOffset = infoGm[base + 2];
    info.hsLen = infoGm[base + 3];
    return info;
}

__aicore__ inline uint32_t GetValidLinkCount(const BInfo &info, uint32_t aNum, uint32_t maxLinkPerB)
{
    if (info.offset >= aNum) {
        return 0;
    }
    const uint32_t remaining = aNum - info.offset;
    return MinU32(MinU32(info.len, maxLinkPerB), remaining);
}

__aicore__ inline void SetWorkspaceBuffers(GM_ADDR workspace,
                                           GlobalTensor<float> &aBlock,
                                           GlobalTensor<float> &bBlock,
                                           GlobalTensor<float> &cBlock,
                                           GlobalTensor<float> &accBlock,
                                           uint32_t slotIdx)
{
    auto base = reinterpret_cast<__gm__ uint8_t *>(GetUserWorkspace(workspace));
    base += static_cast<uint64_t>(slotIdx) * WORKSPACE_BYTES_PER_GROUP;
    aBlock.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(base), A_BLOCK_ELEMS);
    bBlock.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(base + A_BLOCK_BYTES), B_BLOCK_ELEMS);
    cBlock.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(base + A_BLOCK_BYTES + B_BLOCK_BYTES),
                           C_BLOCK_ELEMS);
    accBlock.SetGlobalBuffer(
        reinterpret_cast<__gm__ float *>(base + A_BLOCK_BYTES + B_BLOCK_BYTES + C_BLOCK_BYTES),
        ACC_BLOCK_ELEMS);
}

__aicore__ inline void InitAccumulator(GM_ADDR workspace, TBuf<> &rowBuf,
                                       uint32_t slotIdx, uint32_t localAivIdx,
                                       uint32_t localAivNum)
{
    GlobalTensor<float> aBlock;
    GlobalTensor<float> bBlock;
    GlobalTensor<float> cBlock;
    GlobalTensor<float> accBlock;
    SetWorkspaceBuffers(workspace, aBlock, bBlock, cBlock, accBlock, slotIdx);

    LocalTensor<float> rowLocal = rowBuf.Get<float>();
    Duplicate(rowLocal, 0.0f, ROWS);
    PipeBarrier<PIPE_V>();
    for (uint32_t row = localAivIdx; row < PACK_M; row += localAivNum) {
        DataCopy(accBlock[static_cast<uint64_t>(row) * ROWS], rowLocal, ROWS);
    }
    PipeBarrier<PIPE_ALL>();
    DataCacheCleanAndInvalid<float, CacheLine::ENTIRE_DATA_CACHE>(accBlock);
}

__aicore__ inline void ZeroPackedInputs(GlobalTensor<float> &aBlock,
                                        GlobalTensor<float> &bBlock,
                                        TBuf<> &packRowBuf,
                                        uint32_t localAivIdx,
                                        uint32_t localAivNum)
{
    LocalTensor<float> zeroRow = packRowBuf.Get<float>();
    Duplicate(zeroRow, 0.0f, PACK_K);
    PipeBarrier<PIPE_V>();
    for (uint32_t row = localAivIdx; row < PACK_M; row += localAivNum) {
        DataCopy(aBlock[static_cast<uint64_t>(row) * PACK_K], zeroRow, PACK_K);
    }
    for (uint32_t row = localAivIdx; row < PACK_N; row += localAivNum) {
        DataCopy(bBlock[static_cast<uint64_t>(row) * PACK_K], zeroRow, PACK_K);
    }
    PipeBarrier<PIPE_ALL>();
}

__aicore__ inline void PackOneAntenna(GlobalTensor<float> &hRGm,
                                      GlobalTensor<float> &hIGm,
                                      GlobalTensor<float> &aBlock,
                                      GlobalTensor<float> &bBlock,
                                      TBuf<> &srcRBuf, TBuf<> &srcIBuf,
                                      TBuf<> &scaledRBuf, TBuf<> &scaledIBuf,
                                      TBuf<> &negScaledRBuf,
                                      uint32_t hIdx, uint32_t antenna,
                                      uint32_t linkInChunk, float power)
{
    LocalTensor<float> srcR = srcRBuf.Get<float>();
    LocalTensor<float> srcI = srcIBuf.Get<float>();
    LocalTensor<float> scaledR = scaledRBuf.Get<float>();
    LocalTensor<float> scaledI = scaledIBuf.Get<float>();
    LocalTensor<float> negScaledR = negScaledRBuf.Get<float>();

    const uint64_t srcOffset = static_cast<uint64_t>(hIdx) * H_MATRIX_ELEMS +
                               static_cast<uint64_t>(antenna) * COLS;
    DataCopy(srcR, hRGm[srcOffset], COLS);
    DataCopy(srcI, hIGm[srcOffset], COLS);
    PipeBarrier<PIPE_ALL>();

    Muls(scaledR, srcR, power, COLS);
    Muls(scaledI, srcI, power, COLS);
    Muls(negScaledR, srcR, -power, COLS);
    PipeBarrier<PIPE_V>();

    const uint64_t segment = static_cast<uint64_t>(linkInChunk) * COLS * 2;
    const uint64_t topOffset = static_cast<uint64_t>(antenna) * PACK_K + segment;
    const uint64_t bottomOffset = static_cast<uint64_t>(antenna + ROWS) * PACK_K + segment;
    const uint64_t bOffset = static_cast<uint64_t>(antenna) * PACK_K + segment;

    DataCopy(aBlock[topOffset], scaledR, COLS);
    DataCopy(aBlock[topOffset + COLS], scaledI, COLS);
    DataCopy(aBlock[bottomOffset], scaledI, COLS);
    DataCopy(aBlock[bottomOffset + COLS], negScaledR, COLS);
    DataCopy(bBlock[bOffset], srcR, COLS);
    DataCopy(bBlock[bOffset + COLS], srcI, COLS);
    PipeBarrier<PIPE_ALL>();
}

__aicore__ inline void PackLinkChunk(GM_ADDR hR, GM_ADDR hI, GM_ADDR interfPower,
                                     GM_ADDR workspace, const BInfo &info,
                                     uint32_t aNum, uint32_t maxLinkPerB,
                                     uint32_t chunkStart, uint32_t slotIdx,
                                     uint32_t localAivIdx, uint32_t localAivNum,
                                     TBuf<> &packRowBuf, TBuf<> &srcRBuf,
                                     TBuf<> &srcIBuf, TBuf<> &scaledRBuf,
                                     TBuf<> &scaledIBuf, TBuf<> &negScaledRBuf)
{
    GlobalTensor<float> hRGm;
    GlobalTensor<float> hIGm;
    GlobalTensor<float> aBlock;
    GlobalTensor<float> bBlock;
    GlobalTensor<float> cBlock;
    GlobalTensor<float> accBlock;
    hRGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(hR), static_cast<uint64_t>(aNum) * H_MATRIX_ELEMS);
    hIGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(hI), static_cast<uint64_t>(aNum) * H_MATRIX_ELEMS);
    SetWorkspaceBuffers(workspace, aBlock, bBlock, cBlock, accBlock, slotIdx);

    ZeroPackedInputs(aBlock, bBlock, packRowBuf, localAivIdx, localAivNum);

    const uint32_t linkCount = GetValidLinkCount(info, aNum, maxLinkPerB);
    const uint32_t chunkLinks = MinU32(LINKS_PER_CHUNK, linkCount - chunkStart);
    auto powerGm = reinterpret_cast<__gm__ float *>(interfPower);
    for (uint32_t linkInChunk = 0; linkInChunk < chunkLinks; ++linkInChunk) {
        const uint32_t hIdx = info.offset + chunkStart + linkInChunk;
        const float power = powerGm[hIdx];
        for (uint32_t antenna = localAivIdx; antenna < ROWS; antenna += localAivNum) {
            PackOneAntenna(hRGm, hIGm, aBlock, bBlock,
                           srcRBuf, srcIBuf, scaledRBuf, scaledIBuf, negScaledRBuf,
                           hIdx, antenna, linkInChunk, power);
        }
    }
    PipeBarrier<PIPE_ALL>();
    DataCacheCleanAndInvalid<float, CacheLine::ENTIRE_DATA_CACHE>(aBlock);
    DataCacheCleanAndInvalid<float, CacheLine::ENTIRE_DATA_CACHE>(bBlock);
}

__aicore__ inline void AccumulateChunk(GM_ADDR workspace, uint32_t slotIdx,
                                       uint32_t localAivIdx, uint32_t localAivNum,
                                       TBuf<> &rowBuf, TBuf<> &accBuf)
{
    GlobalTensor<float> aBlock;
    GlobalTensor<float> bBlock;
    GlobalTensor<float> cBlock;
    GlobalTensor<float> accBlock;
    SetWorkspaceBuffers(workspace, aBlock, bBlock, cBlock, accBlock, slotIdx);
    DataCacheCleanAndInvalid<float, CacheLine::ENTIRE_DATA_CACHE>(cBlock);

    LocalTensor<float> cLocal = rowBuf.Get<float>();
    LocalTensor<float> accLocal = accBuf.Get<float>();
    for (uint32_t row = localAivIdx; row < PACK_M; row += localAivNum) {
        DataCopy(cLocal, cBlock[static_cast<uint64_t>(row) * PACK_N], ROWS);
        DataCopy(accLocal, accBlock[static_cast<uint64_t>(row) * ROWS], ROWS);
        PipeBarrier<PIPE_ALL>();
        Add(accLocal, accLocal, cLocal, ROWS);
        PipeBarrier<PIPE_V>();
        DataCopy(accBlock[static_cast<uint64_t>(row) * ROWS], accLocal, ROWS);
        PipeBarrier<PIPE_ALL>();
    }
    DataCacheCleanAndInvalid<float, CacheLine::ENTIRE_DATA_CACHE>(accBlock);
}

__aicore__ inline void WriteRyy(GM_ADDR weakInterf, GM_ADDR ryyR, GM_ADDR ryyI,
                                GM_ADDR workspace, uint32_t bIdx, uint32_t bNum,
                                uint32_t slotIdx, uint32_t localAivIdx,
                                uint32_t localAivNum, TBuf<> &rowBuf, TBuf<> &accBuf)
{
    GlobalTensor<float> aBlock;
    GlobalTensor<float> bBlock;
    GlobalTensor<float> cBlock;
    GlobalTensor<float> accBlock;
    GlobalTensor<float> ryyRGm;
    GlobalTensor<float> ryyIGm;
    SetWorkspaceBuffers(workspace, aBlock, bBlock, cBlock, accBlock, slotIdx);
    ryyRGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(ryyR),
                           static_cast<uint64_t>(bNum) * OUTPUT_MATRIX_ELEMS);
    ryyIGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(ryyI),
                           static_cast<uint64_t>(bNum) * OUTPUT_MATRIX_ELEMS);
    auto weakGm = reinterpret_cast<__gm__ float *>(weakInterf);

    DataCacheCleanAndInvalid<float, CacheLine::ENTIRE_DATA_CACHE>(accBlock);
    LocalTensor<float> realLocal = rowBuf.Get<float>();
    LocalTensor<float> imagLocal = accBuf.Get<float>();
    const uint64_t outputBase = static_cast<uint64_t>(bIdx) * OUTPUT_MATRIX_ELEMS;
    for (uint32_t row = localAivIdx; row < ROWS; row += localAivNum) {
        DataCopy(realLocal, accBlock[static_cast<uint64_t>(row) * ROWS], ROWS);
        DataCopy(imagLocal, accBlock[static_cast<uint64_t>(row + ROWS) * ROWS], ROWS);
        PipeBarrier<PIPE_ALL>();
        realLocal.SetValue(row, realLocal.GetValue(row) + weakGm[static_cast<uint64_t>(bIdx) * ROWS + row]);
        PipeBarrier<PIPE_ALL>();
        DataCopy(ryyRGm[outputBase + static_cast<uint64_t>(row) * ROWS], realLocal, ROWS);
        DataCopy(ryyIGm[outputBase + static_cast<uint64_t>(row) * ROWS], imagLocal, ROWS);
        PipeBarrier<PIPE_ALL>();
    }
    DataCacheCleanAndInvalid<float, CacheLine::ENTIRE_DATA_CACHE>(ryyRGm);
    DataCacheCleanAndInvalid<float, CacheLine::ENTIRE_DATA_CACHE>(ryyIGm);
}

__aicore__ inline void WriteHs(GM_ADDR hR, GM_ADDR hI, GM_ADDR hsR, GM_ADDR hsI,
                               const BInfo &info, uint32_t aNum, uint32_t bIdx,
                               uint32_t bNum, uint32_t localAivIdx,
                               uint32_t localAivNum, TBuf<> &hsRBuf, TBuf<> &hsIBuf)
{
    GlobalTensor<float> hRGm;
    GlobalTensor<float> hIGm;
    GlobalTensor<float> hsRGm;
    GlobalTensor<float> hsIGm;
    hRGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(hR), static_cast<uint64_t>(aNum) * H_MATRIX_ELEMS);
    hIGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(hI), static_cast<uint64_t>(aNum) * H_MATRIX_ELEMS);
    hsRGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(hsR), static_cast<uint64_t>(bNum) * OUTPUT_MATRIX_ELEMS);
    hsIGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(hsI), static_cast<uint64_t>(bNum) * OUTPUT_MATRIX_ELEMS);

    uint32_t hsLen = MinU32(info.hsLen, ROWS);
    if (info.hsOffset >= COLS) {
        hsLen = 0;
    } else {
        hsLen = MinU32(hsLen, COLS - info.hsOffset);
    }
    const bool validSource = info.offset < aNum && hsLen > 0;
    const uint64_t outputBase = static_cast<uint64_t>(bIdx) * OUTPUT_MATRIX_ELEMS;
    LocalTensor<float> hsRLocal = hsRBuf.Get<float>();
    LocalTensor<float> hsILocal = hsIBuf.Get<float>();

    for (uint32_t row = localAivIdx; row < ROWS; row += localAivNum) {
        if (validSource) {
            const uint64_t srcOffset = static_cast<uint64_t>(info.offset) * H_MATRIX_ELEMS +
                                       static_cast<uint64_t>(row) * COLS + info.hsOffset;
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(hsLen * sizeof(float)), 0, 0, 0};
            DataCopyPadExtParams<float> padParams{true, 0, static_cast<uint8_t>(ROWS - hsLen), 0.0f};
            DataCopyPad(hsRLocal, hRGm[srcOffset], copyParams, padParams);
            DataCopyPad(hsILocal, hIGm[srcOffset], copyParams, padParams);
        } else {
            Duplicate(hsRLocal, 0.0f, ROWS);
            Duplicate(hsILocal, 0.0f, ROWS);
        }
        PipeBarrier<PIPE_ALL>();
        const uint64_t dstOffset = outputBase + static_cast<uint64_t>(row) * ROWS;
        DataCopy(hsRGm[dstOffset], hsRLocal, ROWS);
        DataCopy(hsIGm[dstOffset], hsILocal, ROWS);
        PipeBarrier<PIPE_ALL>();
    }
    DataCacheCleanAndInvalid<float, CacheLine::ENTIRE_DATA_CACHE>(hsRGm);
    DataCacheCleanAndInvalid<float, CacheLine::ENTIRE_DATA_CACHE>(hsIGm);
}

__aicore__ inline void RunGemm(GM_ADDR workspace, const TCubeTiling &tiling, uint32_t slotIdx)
{
    using AType = float;
    using BType = float;
    using CType = float;
    using BiasType = float;

    TPipe pipe;
    GlobalTensor<float> aBlock;
    GlobalTensor<float> bBlock;
    GlobalTensor<float> cBlock;
    GlobalTensor<float> accBlock;
    SetWorkspaceBuffers(workspace, aBlock, bBlock, cBlock, accBlock, slotIdx);

    typedef matmul::MatmulType<TPosition::GM, CubeFormat::ND, AType> MatmulAType;
    typedef matmul::MatmulType<TPosition::GM, CubeFormat::ND, BType> MatmulBType;
    typedef matmul::MatmulType<TPosition::GM, CubeFormat::ND, CType> MatmulCType;
    typedef matmul::MatmulType<TPosition::GM, CubeFormat::ND, BiasType> MatmulBiasType;

    matmul::MatmulImpl<MatmulAType, MatmulBType, MatmulCType, MatmulBiasType> mm;
    mm.SetSubBlockIdx(0);
    mm.Init(&tiling, &pipe);
    mm.SetTensorA(aBlock);
    mm.SetTensorB(bBlock, true);
    mm.SetTail(PACK_M, PACK_N);
    mm.SetHF32(false);
    mm.IterateAll(cBlock);
    mm.End();
}
} // namespace CalcRyyKernel

extern "C" __global__ __aicore__ void calc_ryy(GM_ADDR h_r, GM_ADDR h_i,
                                                GM_ADDR h_info, GM_ADDR interf_power,
                                                GM_ADDR weak_interf,
                                                GM_ADDR ryy_r, GM_ADDR ryy_i,
                                                GM_ADDR hs_r, GM_ADDR hs_i,
                                                GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    using namespace CalcRyyKernel;

    auto infoGm = reinterpret_cast<__gm__ uint32_t *>(h_info);
    const uint32_t aNum = tilingData.aNum;
    const uint32_t bNum = tilingData.bNum;
    if (aNum == 0 || bNum == 0 || infoGm[0] != aNum || infoGm[2] != bNum) {
        return;
    }
    const uint32_t maxLinkPerB = MinU32(infoGm[1], aNum);
    const uint32_t linkChunkCount = CeilDiv(maxLinkPerB, LINKS_PER_CHUNK);

    if (TILING_KEY_IS(1)) {
        if (g_coreType == AIV) {
            const uint32_t localAivNum = static_cast<uint32_t>(GetTaskRation());
            if (localAivNum == 0) {
                return;
            }
            const uint32_t aivLinearIdx = static_cast<uint32_t>(GetBlockIdx());
            const uint32_t slotIdx = aivLinearIdx / localAivNum;
            const uint32_t localAivIdx = aivLinearIdx % localAivNum;
            if (slotIdx >= GROUPS_PER_CHUNK) {
                return;
            }

            TPipe pipe;
            TBuf<> packRowBuf;
            TBuf<> srcRBuf;
            TBuf<> srcIBuf;
            TBuf<> scaledRBuf;
            TBuf<> scaledIBuf;
            TBuf<> negScaledRBuf;
            TBuf<> rowBuf;
            TBuf<> accBuf;
            TBuf<> hsRBuf;
            TBuf<> hsIBuf;
            pipe.InitBuffer(packRowBuf, PACK_K * sizeof(float));
            pipe.InitBuffer(srcRBuf, COLS * sizeof(float));
            pipe.InitBuffer(srcIBuf, COLS * sizeof(float));
            pipe.InitBuffer(scaledRBuf, COLS * sizeof(float));
            pipe.InitBuffer(scaledIBuf, COLS * sizeof(float));
            pipe.InitBuffer(negScaledRBuf, COLS * sizeof(float));
            pipe.InitBuffer(rowBuf, ROWS * sizeof(float));
            pipe.InitBuffer(accBuf, ROWS * sizeof(float));
            pipe.InitBuffer(hsRBuf, ROWS * sizeof(float));
            pipe.InitBuffer(hsIBuf, ROWS * sizeof(float));

            for (uint32_t groupBase = 0; groupBase < bNum; groupBase += GROUPS_PER_CHUNK) {
                const uint32_t bIdx = groupBase + slotIdx;
                BInfo info{0, 0, 0, 0};
                uint32_t linkCount = 0;
                if (bIdx < bNum) {
                    info = LoadBInfo(infoGm, bIdx);
                    linkCount = GetValidLinkCount(info, aNum, maxLinkPerB);
                    InitAccumulator(workspace, rowBuf, slotIdx, localAivIdx, localAivNum);
                }

                for (uint32_t chunkIdx = 0; chunkIdx < linkChunkCount; ++chunkIdx) {
                    const uint32_t chunkStart = chunkIdx * LINKS_PER_CHUNK;
                    const bool active = bIdx < bNum && chunkStart < linkCount;
                    if (active) {
                        PackLinkChunk(h_r, h_i, interf_power, workspace, info,
                                      aNum, maxLinkPerB, chunkStart, slotIdx,
                                      localAivIdx, localAivNum, packRowBuf,
                                      srcRBuf, srcIBuf, scaledRBuf, scaledIBuf, negScaledRBuf);
                    }
                    SyncAll<false>();
                    SyncAll<false>();
                    if (active) {
                        AccumulateChunk(workspace, slotIdx, localAivIdx, localAivNum, rowBuf, accBuf);
                    }
                }

                if (bIdx < bNum) {
                    WriteRyy(weak_interf, ryy_r, ryy_i, workspace, bIdx, bNum,
                             slotIdx, localAivIdx, localAivNum, rowBuf, accBuf);
                    WriteHs(h_r, h_i, hs_r, hs_i, info, aNum, bIdx, bNum,
                            localAivIdx, localAivNum, hsRBuf, hsIBuf);
                }
                SyncAll<false>();
            }
            return;
        }

        const uint32_t slotIdx = static_cast<uint32_t>(GetBlockIdx());
        if (slotIdx >= GROUPS_PER_CHUNK) {
            return;
        }
        for (uint32_t groupBase = 0; groupBase < bNum; groupBase += GROUPS_PER_CHUNK) {
            const uint32_t bIdx = groupBase + slotIdx;
            BInfo info{0, 0, 0, 0};
            uint32_t linkCount = 0;
            if (bIdx < bNum) {
                info = LoadBInfo(infoGm, bIdx);
                linkCount = GetValidLinkCount(info, aNum, maxLinkPerB);
            }
            for (uint32_t chunkIdx = 0; chunkIdx < linkChunkCount; ++chunkIdx) {
                const uint32_t chunkStart = chunkIdx * LINKS_PER_CHUNK;
                SyncAll<false>();
                if (bIdx < bNum && chunkStart < linkCount) {
                    RunGemm(workspace, tilingData.cubeTilingData, slotIdx);
                }
                SyncAll<false>();
            }
            SyncAll<false>();
        }
    }
}
