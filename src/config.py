from pathlib import Path

# directories for saving
RESULT_DIR = Path('results')
FIGURE_DIR = Path('figures')
TABLE_DIR = Path('tables')

# colors used for the plots
orange = '#F59729'
ALL_RATTLER_COLOR = {'color':orange,'linestyle':'solid'}
PARTIAL_RATTLER_COLOR = {'color':orange,'linestyle':'dashed'}
HOMOGENOUS_STABLE_COLOR = {'color':'black','linestyle':'solid'}
MIXED_STABLE_COLOR = {'color':'black','linestyle':'dashed'}
AXIS_COLOR = {'color':'grey','linestyle':'solid','lw':"0.5"}
BOUND_COLOR = {'color':'grey','linestyle':'solid','lw':"0.5"}