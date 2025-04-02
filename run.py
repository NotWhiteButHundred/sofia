import argparse
from sparse import COO
from copy import deepcopy
from datetime import timedelta
import numpy as np
import tensorly as tl

import plotly.graph_objects as go

import data
import sofia


parser = argparse.ArgumentParser()


# Model options
parser.add_argument('--R', type=int, default=2)
parser.add_argument('--m', type=int, default=52)
parser.add_argument('--lambda1', type=float, default=0.01)
parser.add_argument('--lambda2', type=float, default=0.01)
parser.add_argument('--lambda3', type=float, default=30.0)
parser.add_argument('--mu', type=float, default=0.3)
parser.add_argument('--phi', type=float, default=0.01)
parser.add_argument('--max_epoch', type=int, default=100)
parser.add_argument('--tol', type=float, default=1e-3)
parser.add_argument('--cycles', type=int, default=2)

# Streaming options
parser.add_argument('--period', type=int, default=3600*24*7)
parser.add_argument('--forecast_steps', type=str, default="13/26/39")

# Dataset options
parser.add_argument('--minmax_scale', action='store_true')
parser.add_argument('--query', type=str, default='vod')
parser.add_argument('--facets', type=str, default='category/brand')
parser.add_argument('--values', type=str, default='None')
parser.add_argument('--start_date', type=str, default='2002-01-01 00:00:00')
parser.add_argument('--end_date', type=str, default='2024-01-01 00:00:00')

# Output options
parser.add_argument('--plot_categories', type=str)

args = parser.parse_args()

events, min_datetime, max_datetime, facets_dict = data.load_as_simple_df(
    query=args.query,
    facets_str=args.facets,
    values=(None if args.values == 'None' else args.values),
    sampling_rate=args.period,
    start_datetime=args.start_date,
    end_datetime=args.end_date)

n_dims = [args.m * args.cycles] + events.iloc[:, 1:-1].nunique().tolist()
print(n_dims)

init_events = events[(events['datetime'] < n_dims[0])].copy()
init_tensor = COO(coords=init_events.iloc[:, :-1].to_numpy().T, data=init_events.iloc[:, -1], shape=n_dims).todense()

if args.minmax_scale:
    norm_events = deepcopy(events)
    norm_value = COO(coords=norm_events.iloc[:, :-1].to_numpy().T, data=norm_events.iloc[:, -1], shape=[norm_events['datetime'].max() + 1] + n_dims[1:]).data.max()
    init_tensor /= norm_value

model = sofia.SOFIA(args.R, args.m, args.lambda1, args.lambda2, args.lambda3, args.mu, args.phi, args.tol)

model.initialize(init_tensor, args.max_epoch)


lss = [int(ls) for ls in args.forecast_steps.split("/")]

pr = args.period
wd = args.m * args.cycles
ls = lss[-1]

plot_categories = args.plot_categories.split("/")

afe = []
fes = [[] for _ in lss]


preds = []
curs = []
futures = []


for t in range(events['datetime'].max() + 1 - (wd + ls)):
#for t in range(10):
    s = t
    e = t + wd

    future_events = events[(e <= events['datetime']) & (events['datetime'] < e + ls)].copy()
    future_events['datetime'] = future_events['datetime'] - e
    future_tensor = COO(coords=future_events.iloc[:, :-1].to_numpy().T,
                        data=future_events.iloc[:, -1], shape=[ls] + n_dims[1:]).todense()
    if args.minmax_scale:
        future_tensor /= norm_value
    future = future_tensor[:, facets_dict[0][plot_categories[0]], facets_dict[1][plot_categories[1]]]
    futures.append(go.Scatter(
        x=np.array([min_datetime + timedelta(seconds=((e + i) * pr)) for i in range(ls)]),
        y=future,
        name="future",
        line=dict(color="red", width=3),
    ))

    pred_tensor = model.predict(ls)
    pred = pred_tensor[:, facets_dict[0][plot_categories[0]], facets_dict[1][plot_categories[1]]]
    preds.append(go.Scatter(
        x=np.array([min_datetime + timedelta(seconds=((e + i) * pr)) for i in range(ls)]),
        y=pred,
        name="pred",
        line=dict(color="blue", width=3),
    ))

    afe.append(np.mean([tl.norm(future_tensor - pred_tensor) / tl.norm(future_tensor) for i in range(ls)]))
    for i, fe in enumerate(fes):
        fe.append(tl.norm(future_tensor[lss[i]-1] - pred_tensor[lss[i]-1]) / tl.norm(future_tensor[lss[i]-1]))

    cur_events = events[(s <= events['datetime']) & (events['datetime'] < e)].copy()
    cur_events['datetime'] = cur_events['datetime'] - s
    cur_tensor = COO(coords=cur_events.iloc[:, :-1].to_numpy().T,
                     data=cur_events.iloc[:, -1], shape=n_dims).todense()
    if args.minmax_scale:
        cur_tensor /= norm_value
    cur = cur_tensor[:, facets_dict[0][plot_categories[0]], facets_dict[1][plot_categories[1]]]
    curs.append(go.Scatter(
        x=np.array([min_datetime + timedelta(seconds=((s + i) * pr)) for i in range(wd)]),
        y=cur,
        name="cur",
        line=dict(color="gray", width=3),
    ))

    next_sub_tensor = future_tensor[0]
    if args.minmax_scale:
        next_sub_tensor /= norm_value

    model.dynamic_update(next_sub_tensor)


steps = []
for s in range(len(preds)):
    cur_visible, future_visible, pred_visible = [False] * len(preds), [False] * len(preds), [False] * len(preds)
    cur_visible[s], future_visible[s], pred_visible[s] = True, True, True
    step = dict(method="update", args=[{"visible": cur_visible + future_visible + pred_visible}])
    steps.append(step)
sliders = [dict(active=0, steps=steps)]
fig = go.Figure(
    data=curs + futures + preds,
    layout=go.Layout(
        sliders=sliders
    )
)
fig.write_html(f"out.html", full_html=True, include_plotlyjs=True)