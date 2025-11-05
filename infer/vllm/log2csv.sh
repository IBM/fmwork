#!/bin/bash
dir=${1}

for f in $(find ${dir} | sort | grep runner.log); do
    d=$(dirname ${f})

    if echo ${d} | grep -q /direct/; then
        if grep -q "FMWORK GEN" ${d}/driver.log; then
            fmw=$(grep -h "FMWORK GEN" ${d}/driver.log)
            ttft=$(echo ${fmw} | cut -d ' ' -f 14)
            itl=$(echo ${fmw} | cut -d ' ' -f 15)
            echo ${d} OK - - ${ttft} ${itl}
            continue
        fi
        target=driver.log
    fi

    if echo ${d} | grep -q /server/; then
        if [[ -f ${d}/client.log && -f ${d}/metrics.log ]]; then
            if [[ -f ${d}/metrics.log ]]; then
                count=$(grep "vllm:request_prefill_time_seconds_count" ${d}/metrics.log 2> /dev/null | cut -d ' ' -f 2)
                sum=$(grep "vllm:request_prefill_time_seconds_sum"   ${d}/metrics.log 2> /dev/null | cut -d ' ' -f 2)
                ttft=$(awk "BEGIN {printf \"%.3f\", ${sum} / ${count}}" 2> /dev/null)
            fi
            [[ -z ${ttft} ]] && ttft="-"

            e2e=$(echo $(grep "Benchmark duration" ${d}/client.log) | rev | cut -d ' ' -f 1 | rev)
            tft=$(echo $(grep "Median TTFT (ms)"   ${d}/client.log) | rev | cut -d ' ' -f 1 | rev)
            itl=$(echo $(grep "Median ITL (ms)"    ${d}/client.log) | rev | cut -d ' ' -f 1 | rev)

            echo ${d} OK ${e2e} ${ttft} ${tft} ${itl}
            continue
        fi
        target=server.log
    fi

    if   grep -q "terminate called after throwing an instance of 'std::out_of_range'" ${d}/${target}; then echo ${d} OOR
    elif grep -q "DtException: Must find space in DDR"                                ${d}/${target}; then echo ${d} MFS
    elif grep -q "DtException: Unable to map graph within architecture constraints"   ${d}/${target}; then echo ${d} UMG
    elif grep -q "DtException: Program verification failed"                           ${d}/${target}; then echo ${d} PVF
    elif grep -q "DtException: Need to find a valid memory space"                     ${d}/${target}; then echo ${d} VMS
    elif grep -q "RuntimeError.*DDR init retried"                                     ${d}/${target}; then echo ${d} DIR
    elif grep -q "TimeoutError: RPC call to execute_model timed out."                 ${d}/${target}; then echo ${d} RPC
    elif grep -q "assert prompt_len <= self\.tkv"                                     ${d}/${target}; then echo ${d} PLT
    elif grep -q "Please reduce the length of the messages or completion"             ${d}/${target}; then echo ${d} CTL
    elif grep -q "assert req_index is not None"                                       ${d}/${target}; then echo ${d} REQ
    elif grep -q "Failed to compile graphs: compile_graph failed"                     ${d}/${target}; then echo ${d} CGF
    else                                                                                                   echo ${d} ...
    fi
done | tr '/ ' ','
