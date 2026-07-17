from sweep import run
print("=== conv_z x contra_wz @ blend=0.2 rev_w=10 ===")
run({"conv_z":[0.15,0.2,0.25],"contra_wz":[40,60,80]},
    dict(half_life=2000, blend=0.2, rev_w=10))
