"""Plot numerical TEA summaries without rerunning the analysis."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from portable_io import validate_output
def summary_figure(inputs, destination, mode):
    destination = validate_output(destination)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import pandas as pd
    from tea_figure import COLORS, LABELS, METHODS

    data = pd.read_csv(inputs / 'method_summary.csv').set_index(['metric', 'method'])
    figure, axes = plt.subplots(1, 3, figsize=(12, 4.7), constrained_layout=True)
    for axis, metric, label in zip(axes, ('cost_USD', 'energy_kWh', 'co2_kg'),
                                  ('Cost (USD/tonne)', 'Electricity (kWh/tonne)', 'CO2 (kg/tonne)')):
        for index, method in enumerate(METHODS):
            row = data.loc[(metric, method)]
            axis.errorbar(row['median'], index,
                          xerr=[[row['median'] - row.q025], [row.q975 - row['median']]],
                          color=COLORS[method], fmt='o', capsize=3)
        axis.set_yticks(range(len(METHODS)), [LABELS[x] for x in METHODS] if axis is axes[0] else [])
        axis.invert_yaxis()
        axis.set_xlabel(label)
        axis.grid(axis='x', alpha=.2)
    count = int(data.n_draws.iloc[0])
    figure.suptitle(f'{mode.capitalize()}: {count:,} paired draws; medians and 95% simulation intervals')
    destination.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination / 'summary.png', dpi=180)
    plt.close(figure)


if __name__=='__main__':
    summary_figure(Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3])
