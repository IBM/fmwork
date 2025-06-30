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
    args = parser.parse_args()

    print()
    results = []

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
    etim = ii = oo = bb = tp = warmup = setup = ttft = itl = thp = None
    engine = "vllm/v1"

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

    for l in s.splitlines():
        l = l.strip()
        split = l.split()

        if 'Warmup took' in l:
            warmup = l.split('Warmup took ')[1].split(' ')[0]
            warmup = re.sub(r'[a-zA-Z]', '', warmup)
            warmup = float(warmup)

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

    model_name = mm
    precision = mv

    result = {
        'timestamp':  etim,
        'metadata_id': args.metadata_id,
        'engine':     engine,
        'model':      model_name,
        'precision':  precision,
        'input':      ii,
        'output':     oo,
        'batch':      bb,
        'tp':         tp,
        'opts':       args.opts,
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
