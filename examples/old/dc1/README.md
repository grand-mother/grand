# DC1 analysis scripts (archived)

Two display scripts for the 10-second data taken at Nançay in July 2022,
written by grand-oma (Olivier Martineau) and committed in January 2023 on the
`beta_dc1` branch. Kept here, with their history, when that branch was merged
on 2026-09-24.

- `ADanalysis.py`: amplitude distributions, trace by trace, for a run file.
  Usage: `python ADanalysis.py <file.root>`.
- `TDAnalysis.py`: time distributions for one detector unit.
  Usage: `python TDAnalysis.py <file.root> <du_id> [n_events_to_show]`.

## They do not run as they are

They use the API from before the 2023 rename, which no longer exists. To
port them:

| Written for | Today |
|---|---|
| `import grand.io.root_trees as rt` | `import grand.dataio as rt` |
| `rt.ADCEventTree(f)` | `rt.TADC(f)` |
| `evt.adc_samples_count_channel0` (and `1`, `2`) | `evt.adc_samples_count_ch` |
| `evt.trace_0` (and `1`, `2`) | `evt.trace_ch` |

`adc_sampling_frequency`, `du_id`, `du_seconds`, `gps_time`,
`get_list_of_events` and `get_event` kept their names. The two renamed fields
now hold all channels together instead of one field per channel, so the
indexing changes too. Not ported or tested here: no DC1 file is in the
repository.
