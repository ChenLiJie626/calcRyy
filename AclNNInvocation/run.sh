#!/bin/bash
set -u

CURRENT_DIR=$(cd "$(dirname "${BASH_SOURCE:-$0}")" && pwd)
A_NUM=${A_NUM:-40}
B_NUM=${B_NUM:-19}
MAX_LINK_PER_B=${MAX_LINK_PER_B:-11}
DEVICE_ID=${DEVICE_ID:-0}
WARMUP_COUNT=${WARMUP_COUNT:-0}
REPEAT_COUNT=${REPEAT_COUNT:-1}
CHECK_EACH_RUN=${CHECK_EACH_RUN:-1}
FIXED_LINK_COUNT=${FIXED_LINK_COUNT:-}
CONTIGUOUS_LINKS=${CONTIGUOUS_LINKS:-0}
ASCEND_TOOLKIT_PATH=${ASCEND_INSTALL_PATH:-${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}}

source "$ASCEND_TOOLKIT_PATH/bin/setenv.bash"
export DDK_PATH=$ASCEND_TOOLKIT_PATH
export NPU_HOST_LIB=$ASCEND_TOOLKIT_PATH/$(arch)-$(uname -s | tr '[:upper:]' '[:lower:]')/lib64

main()
{
    cd "$CURRENT_DIR" || return 1
    GEN_ARGS=(--a-num "$A_NUM" --b-num "$B_NUM" --max-link-per-b "$MAX_LINK_PER_B")
    if [ -n "$FIXED_LINK_COUNT" ]; then
        GEN_ARGS+=(--fixed-link-count "$FIXED_LINK_COUNT")
    fi
    if [ "$CONTIGUOUS_LINKS" = "1" ]; then
        GEN_ARGS+=(--contiguous-links)
    fi
    python3 scripts/gen_data.py "${GEN_ARGS[@]}" || return 1

    cmake -S src -B build -DCMAKE_SKIP_RPATH=TRUE || return 1
    cmake --build build -j16 || return 1

    export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:$ASCEND_TOOLKIT_PATH/opp/vendors/customize/op_api/lib:$NPU_HOST_LIB:$ASCEND_TOOLKIT_PATH/$(arch)-$(uname -s | tr '[:upper:]' '[:lower:]')/devlib:${LD_LIBRARY_PATH:-}
    cd "$CURRENT_DIR/output" || return 1
    ./execute_calc_ryy_op "$A_NUM" "$B_NUM" "$MAX_LINK_PER_B" "$DEVICE_ID" \
        "$WARMUP_COUNT" "$REPEAT_COUNT" "$CHECK_EACH_RUN" || return 1

    cd "$CURRENT_DIR" || return 1
    python3 scripts/verify_result.py --b-num "$B_NUM"
}

main
