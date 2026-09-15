#include "calc_ryy_tiling.h"

#include "exe_graph/runtime/infer_datatype_context.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>

using namespace CalcRyyConst;
using namespace matmul_tiling;

namespace {
#define CALC_RYY_LOG_FAIL(stage, fmt, ...) \
    std::fprintf(stderr, "[CalcRyy][%s] " fmt "\n", stage, ##__VA_ARGS__)

void LogShape(const char *stage, const char *name, const gert::Shape &shape)
{
    std::fprintf(stderr, "[CalcRyy][%s] %s rank=%zu dims=[", stage, name, shape.GetDimNum());
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        std::fprintf(stderr, "%s%ld", i == 0 ? "" : ",", static_cast<long>(shape.GetDim(i)));
    }
    std::fprintf(stderr, "]\n");
}

bool IsDynamic(int64_t dim)
{
    return dim < 0;
}

bool SameOrDynamic(int64_t lhs, int64_t rhs)
{
    return lhs == rhs || IsDynamic(lhs) || IsDynamic(rhs);
}

bool PositiveOrDynamic(int64_t dim)
{
    return dim > 0 || IsDynamic(dim);
}

bool CheckInputShapes(const gert::Shape &hRShape, const gert::Shape &hIShape,
                      const gert::Shape &infoShape, const gert::Shape &powerShape,
                      const gert::Shape &weakShape, bool requireStatic,
                      int64_t &aNum, int64_t &bNum, const char *stage)
{
    if (hRShape.GetDimNum() != 3 || !PositiveOrDynamic(hRShape.GetDim(0)) ||
        !SameOrDynamic(hRShape.GetDim(1), ROWS) || !SameOrDynamic(hRShape.GetDim(2), COLS)) {
        CALC_RYY_LOG_FAIL(stage, "invalid h_r shape, expect [aNum,%d,%d]", ROWS, COLS);
        LogShape(stage, "h_r", hRShape);
        return false;
    }
    if (requireStatic && IsDynamic(hRShape.GetDim(0))) {
        CALC_RYY_LOG_FAIL(stage, "h_r dim0 must be concrete at tiling time");
        return false;
    }
    aNum = hRShape.GetDim(0);

    if (hIShape.GetDimNum() != 3 ||
        !SameOrDynamic(hIShape.GetDim(0), aNum) ||
        !SameOrDynamic(hIShape.GetDim(1), ROWS) ||
        !SameOrDynamic(hIShape.GetDim(2), COLS)) {
        CALC_RYY_LOG_FAIL(stage, "h_i shape must match h_r");
        LogShape(stage, "h_i", hIShape);
        return false;
    }

    if (powerShape.GetDimNum() != 1 || !SameOrDynamic(powerShape.GetDim(0), aNum)) {
        CALC_RYY_LOG_FAIL(stage, "interf_power shape must be [aNum]");
        LogShape(stage, "interf_power", powerShape);
        return false;
    }

    if (weakShape.GetDimNum() != 2 || !PositiveOrDynamic(weakShape.GetDim(0)) ||
        !SameOrDynamic(weakShape.GetDim(1), ROWS)) {
        CALC_RYY_LOG_FAIL(stage, "weak_interf shape must be [bNum,%d]", ROWS);
        LogShape(stage, "weak_interf", weakShape);
        return false;
    }
    if (requireStatic && IsDynamic(weakShape.GetDim(0))) {
        CALC_RYY_LOG_FAIL(stage, "weak_interf dim0 must be concrete at tiling time");
        return false;
    }
    bNum = weakShape.GetDim(0);

    if (infoShape.GetDimNum() != 1) {
        CALC_RYY_LOG_FAIL(stage, "h_info must be one-dimensional");
        LogShape(stage, "h_info", infoShape);
        return false;
    }
    const int64_t infoLen = infoShape.GetDim(0);
    if (!IsDynamic(infoLen)) {
        if (infoLen < 7 || (infoLen - 3) % 4 != 0) {
            CALC_RYY_LOG_FAIL(stage, "h_info length must equal 3 + 4 * bNum, actual %ld",
                              static_cast<long>(infoLen));
            return false;
        }
        const int64_t infoBNum = (infoLen - 3) / 4;
        if (!SameOrDynamic(infoBNum, bNum)) {
            CALC_RYY_LOG_FAIL(stage, "h_info and weak_interf disagree on bNum");
            LogShape(stage, "h_info", infoShape);
            LogShape(stage, "weak_interf", weakShape);
            return false;
        }
    } else if (requireStatic) {
        CALC_RYY_LOG_FAIL(stage, "h_info length must be concrete at tiling time");
        return false;
    }
    return true;
}

bool CheckOutputShape(const gert::Shape &shape, int64_t bNum)
{
    return shape.GetDimNum() == 3 &&
           SameOrDynamic(shape.GetDim(0), bNum) &&
           SameOrDynamic(shape.GetDim(1), ROWS) &&
           SameOrDynamic(shape.GetDim(2), ROWS);
}

static ge::graphStatus InferShape(gert::InferShapeContext *context)
{
    constexpr const char *stage = "InferShape";
    if (context == nullptr) {
        CALC_RYY_LOG_FAIL(stage, "context is nullptr");
        return ge::GRAPH_FAILED;
    }
    const gert::Shape *hRShape = context->GetInputShape(0);
    const gert::Shape *hIShape = context->GetInputShape(1);
    const gert::Shape *infoShape = context->GetInputShape(2);
    const gert::Shape *powerShape = context->GetInputShape(3);
    const gert::Shape *weakShape = context->GetInputShape(4);
    if (hRShape == nullptr || hIShape == nullptr || infoShape == nullptr ||
        powerShape == nullptr || weakShape == nullptr) {
        CALC_RYY_LOG_FAIL(stage, "one or more input shape pointers are null");
        return ge::GRAPH_FAILED;
    }

    int64_t aNum = -1;
    int64_t bNum = -1;
    if (!CheckInputShapes(*hRShape, *hIShape, *infoShape, *powerShape, *weakShape,
                          false, aNum, bNum, stage)) {
        return ge::GRAPH_FAILED;
    }

    for (size_t outputIdx = 0; outputIdx < 4; ++outputIdx) {
        gert::Shape *outputShape = context->GetOutputShape(outputIdx);
        if (outputShape == nullptr) {
            CALC_RYY_LOG_FAIL(stage, "output shape %zu is nullptr", outputIdx);
            return ge::GRAPH_FAILED;
        }
        *outputShape = gert::Shape({bNum, ROWS, ROWS});
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    constexpr const char *stage = "InferDataType";
    if (context == nullptr) {
        CALC_RYY_LOG_FAIL(stage, "context is nullptr");
        return ge::GRAPH_FAILED;
    }
    if (context->GetInputDataType(0) != ge::DT_FLOAT ||
        context->GetInputDataType(1) != ge::DT_FLOAT ||
        context->GetInputDataType(2) != ge::DT_UINT32 ||
        context->GetInputDataType(3) != ge::DT_FLOAT ||
        context->GetInputDataType(4) != ge::DT_FLOAT) {
        CALC_RYY_LOG_FAIL(stage, "input dtypes must be float,float,uint32,float,float");
        return ge::GRAPH_FAILED;
    }
    for (size_t outputIdx = 0; outputIdx < 4; ++outputIdx) {
        if (context->SetOutputDataType(outputIdx, ge::DT_FLOAT) != ge::GRAPH_SUCCESS) {
            CALC_RYY_LOG_FAIL(stage, "SetOutputDataType failed for output %zu", outputIdx);
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}
} // namespace

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    constexpr const char *stage = "TilingFunc";
    if (context == nullptr) {
        CALC_RYY_LOG_FAIL(stage, "context is nullptr");
        return ge::GRAPH_FAILED;
    }

    auto hRTensor = context->GetInputTensor(0);
    auto hITensor = context->GetInputTensor(1);
    auto infoTensor = context->GetInputTensor(2);
    auto powerTensor = context->GetInputTensor(3);
    auto weakTensor = context->GetInputTensor(4);
    if (hRTensor == nullptr || hITensor == nullptr || infoTensor == nullptr ||
        powerTensor == nullptr || weakTensor == nullptr) {
        CALC_RYY_LOG_FAIL(stage, "one or more input tensors are null");
        return ge::GRAPH_FAILED;
    }

    int64_t aNum64 = -1;
    int64_t bNum64 = -1;
    if (!CheckInputShapes(hRTensor->GetOriginShape(), hITensor->GetOriginShape(),
                          infoTensor->GetOriginShape(), powerTensor->GetOriginShape(),
                          weakTensor->GetOriginShape(), true, aNum64, bNum64, stage)) {
        return ge::GRAPH_FAILED;
    }
    for (size_t outputIdx = 0; outputIdx < 4; ++outputIdx) {
        auto outputStorageShape = context->GetOutputShape(outputIdx);
        if (outputStorageShape == nullptr ||
            !CheckOutputShape(outputStorageShape->GetOriginShape(), bNum64)) {
            CALC_RYY_LOG_FAIL(stage, "invalid output shape %zu, expect [%ld,%d,%d]",
                              outputIdx, static_cast<long>(bNum64), ROWS, ROWS);
            return ge::GRAPH_FAILED;
        }
    }

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    MultiCoreMatmulTiling cubeTiling(ascendcPlatform);
    cubeTiling.SetDim(1);
    cubeTiling.SetAType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    cubeTiling.SetBType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    cubeTiling.SetCType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    cubeTiling.SetBiasType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    cubeTiling.SetShape(PACK_M, PACK_N, PACK_K);
    cubeTiling.SetOrgShape(PACK_M, PACK_N, PACK_K);
    cubeTiling.EnableBias(false);
    cubeTiling.SetBufferSpace(-1, -1, -1);

    CalcRyyTilingData tiling;
    if (cubeTiling.GetTiling(tiling.cubeTilingData) == -1) {
        CALC_RYY_LOG_FAIL(stage, "cube tiling failed, m=%d, n=%d, k=%d", PACK_M, PACK_N, PACK_K);
        return ge::GRAPH_FAILED;
    }
    tiling.set_aNum(static_cast<uint32_t>(aNum64));
    tiling.set_bNum(static_cast<uint32_t>(bNum64));

    context->SetBlockDim(DEFAULT_BLOCK_DIM);
    context->SetTilingKey(1);
    auto rawTilingData = context->GetRawTilingData();
    if (rawTilingData == nullptr) {
        CALC_RYY_LOG_FAIL(stage, "raw tiling data is nullptr");
        return ge::GRAPH_FAILED;
    }
    tiling.SaveToBuffer(rawTilingData->GetData(), rawTilingData->GetCapacity());
    rawTilingData->SetDataSize(tiling.GetDataSize());

    size_t *workspaceSizes = context->GetWorkspaceSizes(1);
    if (workspaceSizes == nullptr) {
        CALC_RYY_LOG_FAIL(stage, "workspace size pointer is nullptr");
        return ge::GRAPH_FAILED;
    }
    workspaceSizes[0] = static_cast<size_t>(WORKSPACE_BYTES) + ascendcPlatform.GetLibApiWorkSpaceSize();
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ops {
class CalcRyy : public OpDef {
public:
    explicit CalcRyy(const char *name) : OpDef(name)
    {
        this->SetInferShape(InferShape)
            .SetInferDataType(InferDataType);

        this->Input("h_r").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("h_i").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("h_info").ParamType(REQUIRED).DataType({ge::DT_UINT32}).Format({ge::FORMAT_ND});
        this->Input("interf_power").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Input("weak_interf").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("ryy_r").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("ryy_i").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("hs_r").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});
        this->Output("hs_i").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND});

        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};

OP_ADD(CalcRyy);
} // namespace ops
