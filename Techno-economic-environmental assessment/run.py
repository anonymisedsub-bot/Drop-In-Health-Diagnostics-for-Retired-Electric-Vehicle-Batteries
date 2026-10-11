"""Portable TEA entry point with standardized numerical and visualization outputs."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from portable_io import ROOT, output_paths, validate_output

def execute(script,*arguments):
    subprocess.run([sys.executable,'-B',str(ROOT/script),*map(str,arguments)],check=True)

def default_output(command):
    return output_paths(command)['results']

def summary_figure(inputs,destination,mode):
    execute('visualization/plot_summary.py',inputs,validate_output(destination),mode)

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('demo','full','verify','figures','reference-figures'))
    parser.add_argument('--output',type=Path,help='Custom root containing results/ and visualization/; defaults to this module.')
    parser.add_argument('--draws',type=int,help='Demo draw count; full uses the registered 800,000 draws.')
    parser.add_argument('--skip-panels',action='store_true')
    args=parser.parse_args(argv)
    if args.draws is not None and args.draws<=0: parser.error('--draws must be positive')
    if args.command!='demo' and args.draws is not None: parser.error('--draws applies only to demo; full uses 800000')
    try: paths=output_paths(args.command,args.output)
    except ValueError as exc: parser.error(str(exc))
    results,figures,sources,cache=(paths[k] for k in ('results','figures','figure_data','cache'))
    if args.command in ('figures','reference-figures'):
        execute('visualization/export_tea_panels.py','--input-dir',results,'--cache-dir',cache,
                '--reference-only','--output-dir',figures,'--source-data',sources)
        return 0
    if args.command=='verify':
        execute('verify_analysis.py','--input-dir',results)
        return 0
    results.mkdir(parents=True,exist_ok=True)
    (results/'public_run.json').unlink(missing_ok=True)
    draws=(args.draws or 2000) if args.command=='demo' else 800000
    execute('run_analysis.py','--max-draws',draws,'--output-dir',results)
    execute('verify_analysis.py','--input-dir',results)
    summary_figure(results,figures,args.command)
    if args.command=='full' and not args.skip_panels:
        execute('visualization/build_density_cache.py','--input-dir',results,'--cache-dir',cache)
        execute('visualization/export_tea_panels.py','--input-dir',results,'--cache-dir',cache,
                '--output-dir',figures,'--source-data',sources)
    from portable_io import path_reference
    record=dict(command=args.command,draws=draws,seed=20260921,
                publication_reproduction=args.command=='full',analysis='.',verification='verification.json',
                figure=path_reference(figures/'summary.png',results),
                scope='Incremental testing cost, electricity and electricity-related CO2; not full LCA or NPV.')
    (results/'public_run.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    print(f'Completed {args.command}: {results}',flush=True)
    return 0

if __name__=='__main__':
    try: raise SystemExit(main())
    except subprocess.CalledProcessError as exc: raise SystemExit(exc.returncode)
