#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "acl/acl.h"
#include "common.h"
#include "op_runner.h"

namespace {
constexpr int64_t ROWS = 8;
constexpr int64_t COLS = 32;
constexpr int64_t DEFAULT_A_NUM = 40;
constexpr int64_t DEFAULT_B_NUM = 19;
constexpr int64_t DEFAULT_MAX_LINK_PER_B = 11;
constexpr int DEFAULT_WARMUP_COUNT = 0;
constexpr int64_t DEFAULT_REPEAT_COUNT = 1;
constexpr int DEFAULT_CHECK_EACH_RUN = 1;
} // namespace

bool g_isDevice = false;
int deviceId = 0;
int64_t g_aNum = DEFAULT_A_NUM;
int64_t g_bNum = DEFAULT_B_NUM;
int64_t g_maxLinkPerB = DEFAULT_MAX_LINK_PER_B;

int64_t ParsePositiveIntArg(int argc, char **argv, int argIndex, const char *envName,
                            int64_t defaultValue, const char *name)
{
    const char *value = argc > argIndex ? argv[argIndex] : std::getenv(envName);
    if (value == nullptr) {
        return defaultValue;
    }
    char *end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end == value || parsed <= 0) {
        WARN_LOG("Invalid %s %s, use default %lld", name, value, static_cast<long long>(defaultValue));
        return defaultValue;
    }
    return static_cast<int64_t>(parsed);
}

int64_t ParseNonNegativeIntArg(int argc, char **argv, int argIndex, const char *envName,
                               int64_t defaultValue, const char *name)
{
    const char *value = argc > argIndex ? argv[argIndex] : std::getenv(envName);
    if (value == nullptr) {
        return defaultValue;
    }
    char *end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end == value || parsed < 0) {
        WARN_LOG("Invalid %s %s, use default %lld", name, value, static_cast<long long>(defaultValue));
        return defaultValue;
    }
    return static_cast<int64_t>(parsed);
}

OperatorDesc CreateOpDesc()
{
    std::vector<int64_t> shapeH{g_aNum, ROWS, COLS};
    std::vector<int64_t> shapeInfo{3 + 4 * g_bNum};
    std::vector<int64_t> shapePower{g_aNum};
    std::vector<int64_t> shapeWeak{g_bNum, ROWS};
    std::vector<int64_t> shapeOutput{g_bNum, ROWS, ROWS};
    OperatorDesc opDesc;
    opDesc.AddInputTensorDesc(ACL_FLOAT, shapeH.size(), shapeH.data(), ACL_FORMAT_ND);
    opDesc.AddInputTensorDesc(ACL_FLOAT, shapeH.size(), shapeH.data(), ACL_FORMAT_ND);
    opDesc.AddInputTensorDesc(ACL_UINT32, shapeInfo.size(), shapeInfo.data(), ACL_FORMAT_ND);
    opDesc.AddInputTensorDesc(ACL_FLOAT, shapePower.size(), shapePower.data(), ACL_FORMAT_ND);
    opDesc.AddInputTensorDesc(ACL_FLOAT, shapeWeak.size(), shapeWeak.data(), ACL_FORMAT_ND);
    for (int outputIdx = 0; outputIdx < 4; ++outputIdx) {
        opDesc.AddOutputTensorDesc(ACL_FLOAT, shapeOutput.size(), shapeOutput.data(), ACL_FORMAT_ND);
    }
    return opDesc;
}

bool SetInputData(OpRunner &runner)
{
    size_t fileSize = 0;
    if (!ReadFile("../input/input_h_r.bin", fileSize, runner.GetInputBuffer<void>(0), runner.GetInputSize(0)) ||
        !ReadFile("../input/input_h_i.bin", fileSize, runner.GetInputBuffer<void>(1), runner.GetInputSize(1)) ||
        !ReadFile("../input/input_h_info.bin", fileSize, runner.GetInputBuffer<void>(2), runner.GetInputSize(2)) ||
        !ReadFile("../input/input_interf_power.bin", fileSize, runner.GetInputBuffer<void>(3), runner.GetInputSize(3)) ||
        !ReadFile("../input/input_weak_interf.bin", fileSize, runner.GetInputBuffer<void>(4), runner.GetInputSize(4))) {
        return false;
    }
    INFO_LOG("Set input success");
    return true;
}

bool ProcessOutputData(OpRunner &runner)
{
    const char *paths[] = {
        "../output/output_ryy_r.bin", "../output/output_ryy_i.bin",
        "../output/output_hs_r.bin", "../output/output_hs_i.bin"
    };
    for (size_t outputIdx = 0; outputIdx < 4; ++outputIdx) {
        if (!WriteFile(paths[outputIdx], runner.GetOutputBuffer<void>(outputIdx), runner.GetOutputSize(outputIdx))) {
            return false;
        }
    }
    INFO_LOG("Write output success");
    return true;
}

void DestroyResource()
{
    bool failed = false;
    if (aclrtResetDevice(deviceId) != ACL_SUCCESS) {
        ERROR_LOG("Reset device %d failed", deviceId);
        failed = true;
    }
    if (aclFinalize() != ACL_SUCCESS) {
        ERROR_LOG("Finalize acl failed");
        failed = true;
    }
    INFO_LOG("Destroy resource %s", failed ? "failed" : "success");
}

bool InitResource()
{
    const std::string output = "../output";
    if (access(output.c_str(), 0) == -1 && mkdir(output.c_str(), 0700) != 0) {
        ERROR_LOG("Make output directory failed");
        return false;
    }
    if (aclInit(nullptr) != ACL_SUCCESS) {
        ERROR_LOG("acl init failed");
        return false;
    }
    if (aclrtSetDevice(deviceId) != ACL_SUCCESS) {
        ERROR_LOG("Set device %d failed", deviceId);
        (void)aclFinalize();
        return false;
    }
    aclrtRunMode runMode;
    if (aclrtGetRunMode(&runMode) != ACL_SUCCESS) {
        ERROR_LOG("Get run mode failed");
        DestroyResource();
        return false;
    }
    g_isDevice = runMode == ACL_DEVICE;
    INFO_LOG("Set device[%d], run mode[%d] success", deviceId, runMode);
    return true;
}

bool RunOp(size_t warmupCount, size_t repeatCount, bool checkEveryRun)
{
    OperatorDesc opDesc = CreateOpDesc();
    OpRunner opRunner(&opDesc);
    if (!opRunner.Init() || !SetInputData(opRunner) ||
        !opRunner.RunOp(warmupCount, repeatCount, checkEveryRun) ||
        !ProcessOutputData(opRunner)) {
        return false;
    }
    return true;
}

int main(int argc, char **argv)
{
    g_aNum = ParsePositiveIntArg(argc, argv, 1, "A_NUM", DEFAULT_A_NUM, "aNum");
    g_bNum = ParsePositiveIntArg(argc, argv, 2, "B_NUM", DEFAULT_B_NUM, "bNum");
    g_maxLinkPerB = ParseNonNegativeIntArg(argc, argv, 3, "MAX_LINK_PER_B",
                                           DEFAULT_MAX_LINK_PER_B, "maxLinkPerB");
    deviceId = static_cast<int>(ParseNonNegativeIntArg(argc, argv, 4, "DEVICE_ID", 0, "device id"));
    const int warmupCount = static_cast<int>(ParseNonNegativeIntArg(
        argc, argv, 5, "WARMUP_COUNT", DEFAULT_WARMUP_COUNT, "warmup count"));
    const int64_t repeatCount = ParsePositiveIntArg(
        argc, argv, 6, "REPEAT_COUNT", DEFAULT_REPEAT_COUNT, "repeat count");
    const bool checkEveryRun = ParseNonNegativeIntArg(
        argc, argv, 7, "CHECK_EACH_RUN", DEFAULT_CHECK_EACH_RUN, "check each run") != 0;
    if (g_maxLinkPerB > g_aNum) {
        ERROR_LOG("maxLinkPerB must be <= aNum");
        return FAILED;
    }
    INFO_LOG("Use aNum=%lld, bNum=%lld, maxLinkPerB=%lld, device=%d, warmup=%d, repeats=%lld",
             static_cast<long long>(g_aNum), static_cast<long long>(g_bNum),
             static_cast<long long>(g_maxLinkPerB), deviceId, warmupCount,
             static_cast<long long>(repeatCount));

    if (!InitResource()) {
        return FAILED;
    }
    const bool ok = RunOp(static_cast<size_t>(warmupCount), static_cast<size_t>(repeatCount), checkEveryRun);
    DestroyResource();
    return ok ? SUCCESS : FAILED;
}
