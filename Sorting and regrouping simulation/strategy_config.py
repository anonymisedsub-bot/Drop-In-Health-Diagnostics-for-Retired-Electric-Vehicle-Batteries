"""Current five strategies; stable internal keys align with archived results."""
METHODS=('point','conservative','lower_bound','quadrant_a','quadrant_c')
NAMES={
    'point':'capacity-estimate-only sorting strategy',
    'conservative':'conservative sorting strategy',
    'lower_bound':'lower-bound sorting',
    'quadrant_a':'Uncertainty-guided sorting',
    'quadrant_c':'Uncertainty-guided sorting + targeted RPT',
}
SHORT={
    'point':'Capacity-estimate-only','conservative':'Conservative','lower_bound':'Lower-bound',
    'quadrant_a':'Uncertainty-guided','quadrant_c':'Uncertainty-guided sorting\n+ targeted RPT',
}
IDS={m:f'S{i+1}' for i,m in enumerate(METHODS)}
