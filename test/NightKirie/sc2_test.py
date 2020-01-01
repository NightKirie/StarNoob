scvs = []
idle_scvs = [scv for scv in scvs if scv.order_length == 0]
print(idle_scvs)