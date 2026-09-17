/**
 * @file op_runner.cpp
 *
 * Copyright (C) 2023-2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include "op_runner.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <limits>

#include "acl/acl_op_compiler.h"
#include "aclnn_calc_ryy.h"
#include "common.h"

using namespace std;

extern bool g_isDevice;

namespace {
constexpr int32_t STREAM_TIMEOUT_MS = 5000;
constexpr uint8_t OUTPUT_POISON = 0xA5;

uint64_t HashBuffer(const void *buffer, size_t size)
{
    constexpr uint64_t FNV_OFFSET_BASIS = 14695981039346656037ULL;
    constexpr uint64_t FNV_PRIME = 1099511628211ULL;
    const auto *bytes = reinterpret_cast<const uint8_t *>(buffer);
    uint64_t hash = FNV_OFFSET_BASIS;
    for (size_t i = 0; i < size; ++i) {
        hash ^= bytes[i];
        hash *= FNV_PRIME;
    }
    return hash;
}
} // namespace

OpRunner::OpRunner(OperatorDesc *opDesc) : opDesc_(opDesc)
{
    numInputs_ = opDesc->inputDesc.size();
    numOutputs_ = opDesc->outputDesc.size();
    workspace_ = nullptr;
}

OpRunner::~OpRunner()
{
    if (workspace_!= nullptr) {
        (void)aclrtFree(workspace_);
    }
    
    for (size_t i = 0; i < numInputs_; ++i) {
        (void)aclDestroyTensor(inputTensor_[i]);
        (void)aclDestroyDataBuffer(inputBuffers_[i]);
        (void)aclrtFree(devInputs_[i]);
        if (g_isDevice) {
            (void)aclrtFree(hostInputs_[i]);
        } else {
            (void)aclrtFreeHost(hostInputs_[i]);
        }
    }

    for (size_t i = 0; i < numOutputs_; ++i) {
        (void)aclDestroyTensor(outputTensor_[i]);
        (void)aclDestroyDataBuffer(outputBuffers_[i]);
        (void)aclrtFree(devOutputs_[i]);
        if (g_isDevice) {
            (void)aclrtFree(hostOutputs_[i]);
        } else {
            (void)aclrtFreeHost(hostOutputs_[i]);
        }
    }
}

bool OpRunner::Init()
{
    for (size_t i = 0; i < numInputs_; ++i) {
        auto size = GetInputSize(i);
        const auto allocationSize = std::max<size_t>(size, 1);
        void *devMem = nullptr;
        if (aclrtMalloc(&devMem, allocationSize, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
            ERROR_LOG("Malloc device memory for input[%zu] failed", i);
            return false;
        }
        devInputs_.emplace_back(devMem);
        inputBuffers_.emplace_back(aclCreateDataBuffer(devMem, allocationSize));

        void *hostInput = nullptr;
        if (g_isDevice) {
            if (aclrtMalloc(&hostInput, allocationSize, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
                ERROR_LOG("Malloc device memory for input[%zu] failed", i);
                return false;
            }
        } else {
            if (aclrtMallocHost(&hostInput, allocationSize) != ACL_SUCCESS) {
                ERROR_LOG("Malloc device memory for input[%zu] failed", i);
                return false;
            }
        }
        if (hostInput == nullptr) {
            ERROR_LOG("Malloc memory for input[%zu] failed", i);
            return false;
        }
        hostInputs_.emplace_back(hostInput);

        aclTensor *inputTensor =
            aclCreateTensor(GetInputShape(i).data(), GetInputNumDims(i), GetInputDataType(i), nullptr, 0,
                            GetInputFormat(i), GetInputShape(i).data(), GetInputNumDims(i), devInputs_[i]);
        if (inputTensor == nullptr) {
            ERROR_LOG("Create Tensor for input[%zu] failed", i);
            return false;
        }
        inputTensor_.emplace_back(inputTensor);
    }

    for (size_t i = 0; i < numOutputs_; ++i) {
        auto size = GetOutputSize(i);
        void *devMem = nullptr;
        if (aclrtMalloc(&devMem, size, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
            ERROR_LOG("Malloc device memory for output[%zu] failed", i);
            return false;
        }
        devOutputs_.emplace_back(devMem);
        outputBuffers_.emplace_back(aclCreateDataBuffer(devMem, size));

        void *hostOutput = nullptr;
        if (g_isDevice) {
            if (aclrtMalloc(&hostOutput, size, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
                ERROR_LOG("Malloc device memory for output[%zu] failed", i);
                return false;
            }
        } else {
            if (aclrtMallocHost(&hostOutput, size) != ACL_SUCCESS) {
                ERROR_LOG("Malloc device memory for output[%zu] failed", i);
                return false;
            }
        }
        if (hostOutput == nullptr) {
            ERROR_LOG("Malloc host memory for output[%zu] failed", i);
            return false;
        }
        hostOutputs_.emplace_back(hostOutput);

        aclTensor *outputTensor =
            aclCreateTensor(GetOutputShape(i).data(), GetOutputNumDims(i), GetOutputDataType(i), nullptr, 0,
                            GetOutputFormat(i), GetOutputShape(i).data(), GetOutputNumDims(i), devOutputs_[i]);
        if (outputTensor == nullptr) {
            ERROR_LOG("Create Tensor for output[%zu] failed", i);
            return false;
        }
        outputTensor_.emplace_back(outputTensor);
    }

    return true;
}

const size_t OpRunner::NumInputs()
{
    return numInputs_;
}

const size_t OpRunner::NumOutputs()
{
    return numOutputs_;
}

const size_t OpRunner::GetInputSize(size_t index) const
{
    if (index >= numInputs_) {
        ERROR_LOG("index out of range. index = %zu, numInputs = %zu", index, numInputs_);
        return 0;
    }

    return aclGetTensorDescSize(opDesc_->inputDesc[index]);
}

const size_t OpRunner::GetInputNumDims(size_t index) const
{
    if (index >= numInputs_) {
        ERROR_LOG("index out of range. index = %zu, numInputs = %zu", index, numInputs_);
        return 0;
    }

    return aclGetTensorDescNumDims(opDesc_->inputDesc[index]);
}

aclDataType OpRunner::GetInputDataType(size_t index) const
{
    if (index >= numInputs_) {
        ERROR_LOG("index out of range. index = %zu, numInputs = %zu", index, numInputs_);
        return ACL_DT_UNDEFINED;
    }

    return aclGetTensorDescType(opDesc_->inputDesc[index]);
}

aclFormat OpRunner::GetInputFormat(size_t index) const
{
    if (index >= numInputs_) {
        ERROR_LOG("index out of range. index = %zu, numInputs = %zu", index, numInputs_);
        return ACL_FORMAT_UNDEFINED;
    }

    return aclGetTensorDescFormat(opDesc_->inputDesc[index]);
}

std::vector<int64_t> OpRunner::GetInputShape(size_t index) const
{
    std::vector<int64_t> ret;
    if (index >= numInputs_) {
        ERROR_LOG("index out of range. index = %zu, numInputs = %zu", index, numInputs_);
        return ret;
    }

    auto desc = opDesc_->inputDesc[index];
    for (size_t i = 0; i < aclGetTensorDescNumDims(desc); ++i) {
        int64_t dimSize;
        if (aclGetTensorDescDimV2(desc, i, &dimSize) != ACL_SUCCESS) {
            ERROR_LOG("get dims from tensor desc failed. dims index = %zu", i);
            ret.clear();
            return ret;
        }
        ret.emplace_back(dimSize);
    }

    return ret;
}

size_t OpRunner::GetOutputSize(size_t index) const
{
    if (index >= numOutputs_) {
        ERROR_LOG("index out of range. index = %zu, numOutputs = %zu", index, numOutputs_);
        return 0;
    }

    return aclGetTensorDescSize(opDesc_->outputDesc[index]);
}

const size_t OpRunner::GetOutputNumDims(size_t index) const
{
    if (index >= numOutputs_) {
        ERROR_LOG("index out of range. index = %zu, numOutputs = %zu", index, numOutputs_);
        return 0;
    }

    return aclGetTensorDescNumDims(opDesc_->outputDesc[index]);
}

aclDataType OpRunner::GetOutputDataType(size_t index) const
{
    if (index >= numOutputs_) {
        ERROR_LOG("index out of range. index = %zu, numOutputs = %zu", index, numOutputs_);
        return ACL_DT_UNDEFINED;
    }

    return aclGetTensorDescType(opDesc_->outputDesc[index]);
}

aclFormat OpRunner::GetOutputFormat(size_t index) const
{
    if (index >= numOutputs_) {
        ERROR_LOG("index out of range. index = %zu, numOutputs = %zu", index, numOutputs_);
        return ACL_FORMAT_UNDEFINED;
    }

    return aclGetTensorDescFormat(opDesc_->outputDesc[index]);
}

std::vector<int64_t> OpRunner::GetOutputShape(size_t index) const
{
    std::vector<int64_t> ret;
    if (index >= numOutputs_) {
        ERROR_LOG("index out of range. index = %zu, numOutputs = %zu", index, numOutputs_);
        return ret;
    }

    auto desc = opDesc_->outputDesc[index];
    for (size_t i = 0; i < aclGetTensorDescNumDims(desc); ++i) {
        int64_t dimSize;
        if (aclGetTensorDescDimV2(desc, i, &dimSize) != ACL_SUCCESS) {
            ERROR_LOG("get dims from tensor desc failed. dims index = %zu", i);
            ret.clear();
            return ret;
        }
        ret.emplace_back(dimSize);
    }
    return ret;
}

size_t OpRunner::GetInputElementCount(size_t index) const
{
    if (index >= opDesc_->inputDesc.size()) {
        ERROR_LOG("index out of range. index = %zu, numInputs = %zu", index, numInputs_);
        return 0;
    }

    return aclGetTensorDescElementCount(opDesc_->inputDesc[index]);
}

size_t OpRunner::GetOutputElementCount(size_t index) const
{
    if (index >= opDesc_->outputDesc.size()) {
        ERROR_LOG("index out of range. index = %zu, numOutputs = %zu", index, numOutputs_);
        return 0;
    }

    return aclGetTensorDescElementCount(opDesc_->outputDesc[index]);
}

bool OpRunner::RunOp(size_t warmupCount, size_t repeatCount, bool checkEveryRun)
{
    if (repeatCount == 0) {
        ERROR_LOG("Repeat count must be greater than zero");
        return false;
    }

    for (size_t i = 0; i < numInputs_; ++i) {
        auto size = GetInputSize(i);
        if (size == 0) {
            INFO_LOG("Skip empty input[%zu] copy", i);
            continue;
        }
        aclrtMemcpyKind kind = ACL_MEMCPY_HOST_TO_DEVICE;
        if (g_isDevice) {
            kind = ACL_MEMCPY_DEVICE_TO_DEVICE;
        }
        if (aclrtMemcpy(devInputs_[i], size, hostInputs_[i], size, kind) != ACL_SUCCESS) {
            ERROR_LOG("Copy input[%zu] failed", i);
            return false;
        }
        INFO_LOG("Copy input[%zu] success", i);
    }

    aclrtStream stream = nullptr;
    if (aclrtCreateStream(&stream) != ACL_SUCCESS) {
        ERROR_LOG("Create stream failed");
        return false;
    }
    INFO_LOG("Create stream success");

    aclrtEvent startEvent = nullptr;
    aclrtEvent endEvent = nullptr;
    aclOpExecutor *handle = nullptr;
    bool executorRepeatable = false;
    auto cleanup = [&]() {
        if (startEvent != nullptr) {
            (void)aclrtDestroyEvent(startEvent);
            startEvent = nullptr;
        }
        if (endEvent != nullptr) {
            (void)aclrtDestroyEvent(endEvent);
            endEvent = nullptr;
        }
        if (executorRepeatable && handle != nullptr) {
            (void)aclDestroyAclOpExecutor(handle);
            handle = nullptr;
        }
        if (stream != nullptr) {
            (void)aclrtDestroyStream(stream);
            stream = nullptr;
        }
    };

    size_t workspaceSize = 0;
    auto ret = aclnnCalcRyyGetWorkspaceSize(inputTensor_[0], inputTensor_[1], inputTensor_[2],
                                             inputTensor_[3], inputTensor_[4],
                                             outputTensor_[0], outputTensor_[1],
                                             outputTensor_[2], outputTensor_[3],
                                             &workspaceSize, &handle);
    if (ret != ACL_SUCCESS) {
        cleanup();
        ERROR_LOG("Get Operator Workspace failed. error code is %d", static_cast<int32_t>(ret));
        return false;
    }
    const bool needsRepeatableExecutor = warmupCount + repeatCount > 1;
    if (needsRepeatableExecutor) {
        ret = aclSetAclOpExecutorRepeatable(handle);
        if (ret != ACL_SUCCESS) {
            (void)aclDestroyAclOpExecutor(handle);
            handle = nullptr;
            cleanup();
            ERROR_LOG("Set aclOpExecutor repeatable failed. error code is %d", static_cast<int32_t>(ret));
            return false;
        }
        executorRepeatable = true;
        INFO_LOG("Set aclOpExecutor repeatable success");
    }
    INFO_LOG("Execute aclnnCalcRyyGetWorkspaceSize success, workspace size %lu", workspaceSize);

    if (workspaceSize != 0) {
        if (aclrtMalloc(&workspace_, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
            ERROR_LOG("Malloc device memory failed");
            cleanup();
            return false;
        }
    }

    auto launchOp = [&](const char *phase, size_t index) -> bool {
        const auto launchRet = aclnnCalcRyy(workspace_, workspaceSize, handle, stream);
        if (launchRet != ACL_SUCCESS) {
            ERROR_LOG("Execute Operator failed during %s[%zu]. error code is %d",
                      phase, index, static_cast<int32_t>(launchRet));
            return false;
        }
        return true;
    };

    for (size_t i = 0; i < warmupCount; ++i) {
        if (!launchOp("warmup", i)) {
            cleanup();
            return false;
        }
    }
    if (warmupCount != 0) {
        ret = aclrtSynchronizeStreamWithTimeout(stream, STREAM_TIMEOUT_MS);
        if (ret != ACL_SUCCESS) {
            ERROR_LOG("Warmup stream synchronization failed. error code is %d", static_cast<int32_t>(ret));
            cleanup();
            return false;
        }
        INFO_LOG("Warmup %zu iteration(s) success", warmupCount);
    }

    if (aclrtCreateEvent(&startEvent) != ACL_SUCCESS || aclrtCreateEvent(&endEvent) != ACL_SUCCESS) {
        cleanup();
        ERROR_LOG("Create event failed");
        return false;
    }

    auto resetOutputs = [&]() -> bool {
        for (size_t i = 0; i < numOutputs_; ++i) {
            const auto size = GetOutputSize(i);
            if (aclrtMemset(devOutputs_[i], size, OUTPUT_POISON, size) != ACL_SUCCESS) {
                ERROR_LOG("Reset output[%zu] failed", i);
                return false;
            }
        }
        return true;
    };

    auto copyOutputs = [&]() -> bool {
        for (size_t i = 0; i < numOutputs_; ++i) {
            const auto size = GetOutputSize(i);
            aclrtMemcpyKind kind = ACL_MEMCPY_DEVICE_TO_HOST;
            if (g_isDevice) {
                kind = ACL_MEMCPY_DEVICE_TO_DEVICE;
            }
            if (aclrtMemcpy(hostOutputs_[i], size, devOutputs_[i], size, kind) != ACL_SUCCESS) {
                ERROR_LOG("Copy output[%zu] failed", i);
                return false;
            }
        }
        return true;
    };

    std::vector<float> elapsedTimes;
    elapsedTimes.reserve(repeatCount);
    std::vector<uint64_t> baselineHashes(numOutputs_, 0);
    for (size_t iteration = 0; iteration < repeatCount; ++iteration) {
        if (checkEveryRun && !resetOutputs()) {
            cleanup();
            return false;
        }
        if (aclrtRecordEvent(startEvent, stream) != ACL_SUCCESS ||
            !launchOp("repeat", iteration) ||
            aclrtRecordEvent(endEvent, stream) != ACL_SUCCESS) {
            ERROR_LOG("Launch or event recording failed at repeat[%zu]", iteration);
            cleanup();
            return false;
        }

        ret = aclrtSynchronizeStreamWithTimeout(stream, STREAM_TIMEOUT_MS);
        if (ret != ACL_SUCCESS) {
            ERROR_LOG("Synchronize stream failed at repeat[%zu]. error code is %d",
                      iteration, static_cast<int32_t>(ret));
            cleanup();
            return false;
        }

        float elapsedMs = 0.0f;
        if (aclrtEventElapsedTime(&elapsedMs, startEvent, endEvent) != ACL_SUCCESS) {
            ERROR_LOG("Get event elapsed time failed at repeat[%zu]", iteration);
            cleanup();
            return false;
        }
        elapsedTimes.push_back(elapsedMs);

        if (checkEveryRun) {
            if (!copyOutputs()) {
                cleanup();
                return false;
            }
            for (size_t outputIdx = 0; outputIdx < numOutputs_; ++outputIdx) {
                const uint64_t hash = HashBuffer(hostOutputs_[outputIdx], GetOutputSize(outputIdx));
                if (iteration == 0) {
                    baselineHashes[outputIdx] = hash;
                } else if (hash != baselineHashes[outputIdx]) {
                    ERROR_LOG("Output[%zu] hash mismatch at repeat[%zu]: expected %llu, got %llu",
                              outputIdx, iteration,
                              static_cast<unsigned long long>(baselineHashes[outputIdx]),
                              static_cast<unsigned long long>(hash));
                    cleanup();
                    return false;
                }
            }
        }
    }

    if (!checkEveryRun && !copyOutputs()) {
        cleanup();
        return false;
    }

    const auto minmax = std::minmax_element(elapsedTimes.begin(), elapsedTimes.end());
    double totalMs = 0.0;
    for (const float elapsedMs : elapsedTimes) {
        totalMs += elapsedMs;
    }
    INFO_LOG("Kernel timing: warmup=%zu repeats=%zu average=%.3f ms min=%.3f ms max=%.3f ms",
             warmupCount, repeatCount, totalMs / static_cast<double>(repeatCount),
             *minmax.first, *minmax.second);
    if (checkEveryRun) {
        INFO_LOG("All %zu repeated output hash checks passed", repeatCount);
    }

    cleanup();
    return true;
}

template <typename T> void DoPrintData(const T *data, size_t count, size_t elementsPerRow)
{
    assert(elementsPerRow != 0);
    for (size_t i = 0; i < count; ++i) {
        std::cout << std::setw(10) << data[i];
        if (i % elementsPerRow == elementsPerRow - 1) {
            std::cout << std::endl;
        }
    }
}

void DoPrintFp16Data(const aclFloat16 *data, size_t count, size_t elementsPerRow)
{
    assert(elementsPerRow != 0);
    for (size_t i = 0; i < count; ++i) {
        std::cout << std::setw(10) << std::setprecision(4) << aclFloat16ToFloat(data[i]);
        if (i % elementsPerRow == elementsPerRow - 1) {
            std::cout << std::endl;
        }
    }
}

void PrintData(const void *data, size_t count, aclDataType dataType, size_t elementsPerRow)
{
    if (data == nullptr) {
        ERROR_LOG("Print data failed. data is nullptr");
        return;
    }

    switch (dataType) {
        case ACL_BOOL:
            DoPrintData(reinterpret_cast<const bool *>(data), count, elementsPerRow);
            break;
        case ACL_INT8:
            DoPrintData(reinterpret_cast<const int8_t *>(data), count, elementsPerRow);
            break;
        case ACL_UINT8:
            DoPrintData(reinterpret_cast<const uint8_t *>(data), count, elementsPerRow);
            break;
        case ACL_INT16:
            DoPrintData(reinterpret_cast<const int16_t *>(data), count, elementsPerRow);
            break;
        case ACL_UINT16:
            DoPrintData(reinterpret_cast<const uint16_t *>(data), count, elementsPerRow);
            break;
        case ACL_INT32:
            DoPrintData(reinterpret_cast<const int32_t *>(data), count, elementsPerRow);
            break;
        case ACL_UINT32:
            DoPrintData(reinterpret_cast<const uint32_t *>(data), count, elementsPerRow);
            break;
        case ACL_INT64:
            DoPrintData(reinterpret_cast<const int64_t *>(data), count, elementsPerRow);
            break;
        case ACL_UINT64:
            DoPrintData(reinterpret_cast<const uint64_t *>(data), count, elementsPerRow);
            break;
        case ACL_FLOAT16:
            DoPrintFp16Data(reinterpret_cast<const aclFloat16 *>(data), count, elementsPerRow);
            break;
        case ACL_FLOAT:
            DoPrintData(reinterpret_cast<const float *>(data), count, elementsPerRow);
            break;
        case ACL_DOUBLE:
            DoPrintData(reinterpret_cast<const double *>(data), count, elementsPerRow);
            break;
        default:
            ERROR_LOG("Unsupported type: %d", dataType);
    }
}

void OpRunner::PrintInput(size_t index, size_t numElementsPerRow)
{
    if (index >= numInputs_) {
        ERROR_LOG("index out of range. index = %zu, numOutputs = %zu", index, numInputs_);
        return;
    }

    auto desc = opDesc_->inputDesc[index];
    PrintData(hostInputs_[index], GetInputElementCount(index), aclGetTensorDescType(desc), numElementsPerRow);
}

void OpRunner::PrintOutput(size_t index, size_t numElementsPerRow)
{
    if (index >= numOutputs_) {
        ERROR_LOG("index out of range. index = %zu, numOutputs = %zu", index, numOutputs_);
        return;
    }

    auto desc = opDesc_->outputDesc[index];
    PrintData(hostOutputs_[index], GetOutputElementCount(index), aclGetTensorDescType(desc), numElementsPerRow);
}
