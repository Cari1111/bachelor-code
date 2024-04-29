import pypsa
import numpy as np

network = pypsa.Network()
print('hi')

#add three buses
n_buses = 3

for i in range(n_buses):
    network.add("Bus", "My bus {}".format(i),  v_nom=20.)

#add three lines in a ring
for i in range(n_buses):
    network.add("Line", "My line {}".format(i),
                bus0="My bus {}".format(i),
                bus1="My bus {}".format((i+1)%3),
                x=0.1,
                r=0.01)

#add a generator at bus 0
network.add("Generator", "My gen",
            bus="My bus 0",
            p_set=100,
            control="PQ")


#add a load at bus 1
network.add("Load", "My load",
            bus="My bus 1",
            p_set=100,
            q_set=100)

print(network.buses)