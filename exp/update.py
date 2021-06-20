import re

import wandb
import json
import re


def get(string):

    return float(slope), float(y_0), float(y_n)


if __name__ == '__main__':

    api = wandb.Api()
    runs = api.runs('n_michlo/weight-init', order='+created_at')

    for i, run in enumerate(runs):
        print(f'updating {i}: {run.name} ...', end='')
        # get new config values
        try:
            [(slope, y_0, y_n)] = re.findall(r't:slope=(-?\d+\.?\d*):beg=(-?\d+\.?\d*):end=(-?\d+\.?\d*)_.+', run.name)
        except:
            print('FAILED!')
            continue
        # update config
        run.config['targ_std_slope'] = float(slope)
        run.config['targ_std_y_0'] = float(y_0)
        run.config['targ_std_y_n'] = float(y_n)
        # update run
        run.update()
        print('SUCCESS!')

# MIN SCORE SUMMARY:
#       52    55    58   61    64
y_0   = 0.01, 0.25, 0.5, 0.75, 0.99
#       57    53    55   56    59
y_n   = 0.01, 0.25, 0.5, 0.75, 0.99
#       53    54    56   55    55    54
slope = 0.999, 0.9, 0.5, -0.5, -0.9, -0.999

# generally good (slope, y_0, y_n)
# 1.  0.9   0.01 0.25
# 2.  0.999 0.01 0.5
# 3. -0.9   0.01 0.25
# 4. -0.5   0.01 0.25
