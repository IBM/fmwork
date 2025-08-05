import argparse
import json
import os
import re
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata_id', type=str, required=True)
    parser.add_argument('--model',       type=str, required=True, help="Hugging Face model ID (e.g., ibm-granite/granite-3.3-8b-instruct), Override model name in output JSON")
    parser.add_argument('--precision',   type=str, required=True, help="Precision string (e.g., bf16)")
    parser.add_argument('--output',      type=str, required=True, help="Output JSON file (list of dicts)")
    parser.add_argument('--debug',       action='store_true')
    parser.add_argument('--opts',        type=str, default="")

    args, extra_files = parser.parse_known_args()

    if not extra_files:
        print("Error: No exp.log files provided via stdin or xargs.")
        sys.exit(1)

    results = []
    for path in extra_files:
        result = process_single_file(path, args)
        if result:
            results.append(result)

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=4)

def process_single_file(path, args):
    etim = ii = oo = bb = tp = warmup = setup = ttft = itl = thp = None
    engine = "vllm/v1"
    spyre_opts = None

    try:
        with open(path, 'r') as f:
            s = f.read()
    except:
        print(f"Error reading file: {path}")
        return

    if 'FMWORK GEN' not in s:
        print('\nError: No GEN --', path)
        return
    if 'DONE' not in s:
        print('\nError: No DONE --', path)
        return

    lines = s.splitlines()
    spyre_start_idx = None

    for i, l in enumerate(lines):
        l = l.strip()
        split = l.split()

        if "init engine" in l and "took" in l and "seconds" in l:
            match = re.search(r'took\s+([0-9.]+)\s+seconds', l)
            if match:
                warmup = float(match.group(1))

        if l.startswith('Settings for Spyre'):
            spyre_start_idx = i

        if l.startswith('FMWORK SETUP'):
            setup = float(split[2])

        if l.startswith('FMWORK GEN'):
            if 'nan' in l:
                print('Error: GEN with nan --', path)
                return
            etim =       split[2]
            ii   =   int(split[3])
            oo   =   int(split[4])
            bb   =   int(split[5])
            tp   =   int(split[6])
            inf  = float(split[7])
            gen  = float(split[8])
            ttft = float(split[9])
            itl  = float(split[10])
            thp  = float(split[11])

    if spyre_start_idx is not None:
        kv_pairs = []
        lines_after = lines[spyre_start_idx + 1:]
        start_idx = None

        for idx, line in enumerate(lines_after):
            if line.strip().startswith('---'):
                start_idx = idx + 1
                break

        if start_idx is not None:
            for line in lines_after[start_idx:]:
                line = line.strip()
                if line.startswith('--'):
                    break
                if not line:
                    continue
                if ':' in line:
                    key, val = map(str.strip, line.split(':', 1))
                    kv_pairs.append(f"{key}:{val}")

        spyre_opts = ','.join(kv_pairs)
    else:
        spyre_opts = ''

    # Merge CLI + Spyre opts, convert to list of strings
    if spyre_opts and args.opts:
        opts_final = spyre_opts + ',' + args.opts
    elif spyre_opts:
        opts_final = spyre_opts
    else:
        opts_final = args.opts

    opts_list = [s.strip() for s in opts_final.split(',') if s.strip()]

    result = {
        'timestamp':  etim,
        'metadata_id': args.metadata_id,
        'engine':     engine,
        'model':      args.model,
        'precision':  args.precision,
        'input':      ii,
        'output':     oo,
        'batch':      bb,
        'tp':         tp,
        'opts':       opts_list,
        'warmup':     warmup,
        'setup':      setup,
        'ttft':       ttft,
        'itl':        itl,
        'thp':        thp,
    }

    print('FMWORK POS',
          ' '.join(map(str, result.values())),
          path)

    if args.debug:
        print()
        print(json.dumps(result, indent=4))

    return result

if __name__ == '__main__':
    main()
