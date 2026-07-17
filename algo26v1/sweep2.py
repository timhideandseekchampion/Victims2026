from sweep import run
run({"blend":[0.15,0.2,0.25,0.3],"rev_w":[8,10,12,15,20]},
    dict(half_life=2000, conv_z=0.2, contra_wz=60))
