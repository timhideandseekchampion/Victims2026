"""
test_batch100_H87.py

DIAGNOSTIC (H87): how often do adjacent-day flip-flops (sign flips, then flips right back within 2
days -- a single-day "spike") occur in the shipped v10 idio book, and would a same-direction "settle"
filter suppressing them plausibly help or hurt? Quick estimate (not a full new position-array test):
for every such spike day, compare the ACTUAL realized $ (pnl - commission) of trading the 1-day spike
vs. the counterfactual of just holding the pre-spike direction straight through (no flip out, no flip
back -- 2 commission events avoided, and the position during the spike day would have been the
opposite sign of what was actually held).
"""
import numpy as np
import batch100_common_gi as B

sanity_ok = B.print_sanity("(shared cache)")

WZ_V10, days, nIdio, nt = B.WZ_V10, B.days, B.nIdio, B.nt
P_, dlr, commRate = B.P_, B.dlr, B.commRate
POS_base = B.POS_BASE

days_arr = np.array(days)
SGN = np.sign(POS_base[1:, :])  # (nIdio, nt) actual traded sign path (0 before warmup)

print("\n=== counting adjacent-day spike flip-flops: sign(t-1)=A, sign(t)=-A, sign(t+1)=A, A!=0, "
      "restricted to day>=500 (OLD/NEW test range) ===")
n_spike = 0
spike_list = []  # (name, day)
for i in range(nIdio):
    s = SGN[i]
    for t in range(500, nt - 1):
        a, b, c = s[t - 1], s[t], s[t + 1]
        if a != 0 and b == -a and c == a:
            n_spike += 1
            spike_list.append((i, t))

total_nameday = nIdio * (nt - 1 - 500)
print(f"  {n_spike} spike flip-flops found across {total_nameday} name-days "
      f"({100 * n_spike / total_nameday:.3f}% of name-days)")

print("\n=== quick $ estimate: actual (flip out + flip back) vs counterfactual (hold steady) for each "
      "spike, using realized next-day price moves and shipped commission rates ===")
actual_total, cf_total = 0.0, 0.0
for i, t in spike_list:
    name_row = i + 1  # offset into P_/dlr (instrument 0 is ALGO)
    cur = dlr[name_row]  # idio names are always sign * full dlr/cur size (same magnitude before/after)
    mag_tm1 = abs(POS_base[name_row, t - 1])
    mag_t = abs(POS_base[name_row, t])
    mag_tp1 = abs(POS_base[name_row, t + 1])
    A = SGN[i, t - 1]
    # realized pnl at the transition into day t (position set at t-1 earns move t-1->t) -- identical
    # under actual and counterfactual (a same-day-t decision cannot retroactively change it), included
    # in both totals for transparency; it cancels exactly in the actual-vs-counterfactual difference.
    pnl_t = POS_base[name_row, t - 1] * (P_[name_row, t] - P_[name_row, t - 1])
    pnl_tp1 = POS_base[name_row, t] * (P_[name_row, t + 1] - P_[name_row, t])
    comm_out = commRate[name_row] * abs(POS_base[name_row, t] - POS_base[name_row, t - 1]) * P_[name_row, t]
    comm_back = commRate[name_row] * abs(POS_base[name_row, t + 1] - POS_base[name_row, t]) * P_[name_row, t + 1]
    actual = pnl_t + pnl_tp1 - comm_out - comm_back

    # counterfactual: hold at A straight through (no flip out, no flip back -> zero extra commission
    # relative to whatever the position would otherwise have been at t+1 anyway, since t+1's sign is
    # already A -- so the ONLY difference is what happens ON day t: position stays at +A*mag instead of
    # flipping to -A*mag, and no commission is paid at t or t+1 for this particular in/out pair)
    cf_pos_t = A * mag_tm1
    cf_pnl_t = POS_base[name_row, t - 1] * (P_[name_row, t] - P_[name_row, t - 1])  # unchanged (held pos same as t-1)
    cf_pnl_tp1 = cf_pos_t * (P_[name_row, t + 1] - P_[name_row, t])
    cf = cf_pnl_t + cf_pnl_tp1  # no flip commission at all under the counterfactual

    actual_total += actual
    cf_total += cf

print(f"  actual (flip out + flip back), summed over {len(spike_list)} spikes: ${actual_total:,.0f}")
print(f"  counterfactual (settle filter: hold steady through the spike): ${cf_total:,.0f}")
print(f"  implied gain from suppressing spikes: ${cf_total - actual_total:,.0f} "
      f"({'settle filter looks helpful' if cf_total > actual_total else 'settle filter looks harmful'} "
      f"on this quick estimate)")
