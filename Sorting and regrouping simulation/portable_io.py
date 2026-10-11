"""Separate numerical results from visualization outputs; protect frozen inputs."""
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent
PACKAGE = ROOT.parent


def validate_output(destination):
    path = Path(destination).resolve()
    allowed = (ROOT/'results/runs', ROOT/'results/figure_data', ROOT/'visualization/outputs', PACKAGE/'runs')
    if path == PACKAGE or path in PACKAGE.parents:
        raise ValueError('Output is protected: choose a new output root.')
    if path.is_relative_to(PACKAGE) and not any(path.is_relative_to(p) for p in allowed):
        raise ValueError('Output is protected: use results/runs, results/figure_data or visualization/outputs.')
    return path


def output_paths(command, output_root=None, *, reference_verify=False):
    base = ROOT if output_root is None else Path(output_root).resolve()
    if base != ROOT:
        # Custom roots mirror the complete results/ and visualization/ structure.
        if base == PACKAGE or base in PACKAGE.parents or (base.is_relative_to(PACKAGE) and not base.is_relative_to(PACKAGE/'runs')):
            raise ValueError('Output root is protected: use the module root, package runs/<name>, or a new external directory.')
    reference = command in ('figures','reference-figures') or (command=='verify' and reference_verify and output_root is None)
    mode = 'full' if command=='verify' else command
    if reference:
        return dict(results=ROOT/'results/reference', figures=base/'visualization/outputs',
                    figure_data=base/'results/figure_data', cache=ROOT/'results/reference/figure_source')
    results = base/'results/runs'/mode
    return dict(results=results, figures=base/'visualization/outputs/runs'/mode,
                figure_data=results/'figure_data', cache=results/'figure_source')


def path_reference(target, base):
    try:
        return Path(os.path.relpath(target, base)).as_posix()
    except ValueError:
        return Path(target).resolve().as_posix()
