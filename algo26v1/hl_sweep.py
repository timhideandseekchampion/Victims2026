from combined_lab import evaluate
print(f"{'HALF_LIFE':>10} {'S@250':>8} {'S@440':>8}")
for hl in [90,120,180,250,375,500,1000,2000]:
    r=evaluate(half_life=hl, conv_z=0.2, blend=0.0, contra_wz=60)
    print(f"{hl:>10} {r[250][0]:8.1f} {r[440][0]:8.1f}")
