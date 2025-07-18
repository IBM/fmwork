import argparse
import json
import os
import re

def main():
    parser = argparse.ArgumentParser()
    parser.add = parser.add_argument
    parser.add('--path',        type=str, required=True)
    parser.add('--metadata_id', type=str, required=True)
    parser.add('--output',      type=str)
    parser.add('--debug',       action='store_true')
    parser.add('--opts',        type=str, default="")
    parser.add('--model', type=str, default=None, help="Hugging Face model ID (e.g., ibm-granite/granite-3.3-8b-instruct), Override model name in output JSON")
    args = parser.parse_args()

    print()
    if args.model:
        print(f"Overriding model name with: {args.model}")
    results = []

    if args.model:
        # Only use a dummy placeholder for mm since it's overridden anyway
        mm = "__model"
        for mv in os.listdir(args.path):
            path_ = os.path.join(args.path, mv)
            if not os.path.isdir(path_):
                continue
            for item in os.listdir(path_):
                path__ = os.path.join(path_, item)
                result = process(path__, args, mm, mv)
                if result:
                    results.append(result)
    else:
        for mm in os.listdir(args.path):
            path_ = os.path.join(args.path, mm)
            if not os.path.isdir(path_):
                continue

            for mv in os.listdir(path_):
                path__ = os.path.join(path_, mv)
                if not os.path.isdir(path__):
                    continue

                for item in os.listdir(path__):
                    path___ = os.path.join(path__, item)
                    result = process(path___, args, mm, mv)
                    if result:
                        results.append(result)

    print()

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=4)

def process(path, args, mm, mv):
    # Initialize variables
    etim = ii = oo = bb = tp = warmup = setup = ttft = itl = thp = None
    engine = "vllm/v1"
    spyre_opts = None

    # Read log file
    try:
        with open(path, 'r') as f:
            s = f.read()
    except:
        print(f"Error reading file: {path}")
        return

    # Basic validation: must contain FMWORK GEN and DONE markers
    if 'FMWORK GEN' not in s:
        print('\nError: No GEN --', path)
        return
    if 'DONE' not in s:
        print('\nError: No DONE --', path)
        return

    # Split log file into lines
    lines = s.splitlines()

    # Parse line by line to extract metadata and locate Spyre config block
    spyre_start_idx = None  # index of the "Settings for Spyre" header
    for i, l in enumerate(lines):
        l = l.strip()
        split = l.split()

        # Detect the start of Spyre config block (just record the index)
        if l.startswith('Settings for Spyre'):
            spyre_start_idx = i

        # Extract warmup time from line like: "Warmup took XXs"
        if 'Warmup took' in l:
            warmup = l.split('Warmup took ')[1].split(' ')[0]
            warmup = re.sub(r'[a-zA-Z]', '', warmup)
            warmup = float(warmup)

        # Extract setup time from: "FMWORK SETUP <time_in_ms>"
        if l.startswith('FMWORK SETUP'):
            setup = float(split[2])

        # Extract metrics from FMWORK GEN line
        if l.startswith('FMWORK GEN'):
            if 'nan' in l:
                print('Error: GEN with nan --', path)
                return
            etim =       split[2]     # timestamp
            ii   =   int(split[3])    # input length
            oo   =   int(split[4])    # output length
            bb   =   int(split[5])    # batch size
            tp   =   int(split[6])    # tensor parallelism
            inf  = float(split[7])    # inference time
            gen  = float(split[8])    # generation time
            ttft = float(split[9])    # time to first token
            itl  = float(split[10])   # iteration latency
            thp  = float(split[11])   # throughput

    # Extract key-value pairs under "Settings for Spyre"
    if spyre_start_idx is not None:
        kv_pairs = []
        lines_after = lines[spyre_start_idx + 1:]
        start_idx = None

        # Find the first dashed line after the title (e.g. "------------------")
        for idx, line in enumerate(lines_after):
            if line.strip().startswith('---'):
                start_idx = idx + 1  # actual config starts after the dashed line
                break

        # Collect key-value pairs until next dashed line (e.g. "--------------")
        if start_idx is not None:
            for line in lines_after[start_idx:]:
                line = line.strip()
                if line.startswith('--'):
                    break  # end of Spyre block
                if not line:
                    continue  # skip blank lines
                if ':' in line:
                    key, val = map(str.strip, line.split(':', 1))
                    kv_pairs.append(f"{key}:{val}")

        spyre_opts = ','.join(kv_pairs)
    else:
        spyre_opts = ''


    # Final opts: combine Spyre settings (if any) with command-line opts
    if spyre_opts and args.opts:
        opts_final = spyre_opts + ',' + args.opts
    elif spyre_opts:
        opts_final = spyre_opts
    else:
        opts_final = args.opts

    # Build result dictionary
    result = {
        'timestamp':  etim,
        'metadata_id': args.metadata_id,
        'engine':     engine,
        'model':      args.model if args.model else mm,
        'precision':  mv,
        'input':      ii,
        'output':     oo,
        'batch':      bb,
        'tp':         tp,
        'opts':       opts_final,
        'warmup':     warmup,
        'setup':      setup,
        'ttft':       ttft,
        'itl':        itl,
        'thp':        thp,
    }

    # Print summary line
    print('FMWORK POS',
          ' '.join(map(str, result.values())),
          path)

    # If debug mode is on, print JSON-formatted result
    if args.debug:
        print()
        print(json.dumps(result, indent=4))

    return result

if __name__ == '__main__':
    main()
