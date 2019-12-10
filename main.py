from antcolony import AntColony
import matplotlib.pyplot as plt
import numpy as np

gen_index = 2

i = gen_index - 1

load = np.array([0, 150, 200, 250, 180])  # MW each hour
p_g = np.array([0, 20, 25, 30, 22])  # $/MWh each hour
p_r = p_g + np.array([0, 2, 3, 4, 2])  # $/MW each hour
p_n = p_g + np.array([0, 1, 2, 3, 3])  # $/MW each hour

ramp_rate = np.array([120, 60, 60])  # MW/h each unit
quick_start = np.array([0, 20, 20])  # MW each unit
minOn = np.array([2, 2, 1])  # Minimum on time in hours
minOff = np.array([2, 1, 1])  # Minimum off time in hours
initialState = np.array([1, 0, 0])  # (1 means it starts on)
initialHour = np.array([4, 2, 1])  # hours it has been in the initial state
state = np.array([])
Pg_min = np.array([50, 20, 20])  # MW each unit
Pg_max = np.array([200, 100, 100])  # MW each unit
p_fuel = np.array([2, 2, 2.5])  # $/MBtu each unit
startup = p_fuel*np.array([200, 25, 25])  # $ to start the unit
shutdown = np.array([0, 0, 0])  # $ to shutdown unit
a = p_fuel*np.array([400, 25, 25])  # each unit
b = p_fuel*np.array([8, 10, 10])  # each unit
c = p_fuel*np.array([0.01, 0.025, 0.02])  # each unit
P0 = np.array([
    [0, 0, 0, 0, 0],
    [0, 30, 30, 30, 30],
    [0, 0, 0, 0, 0]])  # MW each hour
k = 1  # markup percent for bilateral contract
# Contract price for gen2
f_bl = k*(np.array([a]).T*np.array([[0], [1], [0]]) + np.array([b]).T*P0 + np.array([c]).T*P0**2)
lg = 0
lr = 0
ln = 0
n_hours = 4
cost_on = np.zeros(n_hours + 1)
cost_off = np.zeros(n_hours + 1)
r = np.zeros(n_hours + 1)
n = np.zeros(n_hours + 1)
p = np.zeros(n_hours + 1)
L = np.zeros(n_hours + 1)
n_off = np.zeros(n_hours + 1)
for t in range(1, n_hours + 1):
    # Unit On
    if p_r[t] > p_n[t]:
        Rmin = 0
        Rmax = ramp_rate[i] / 60 * 10  # ten minute ramp capacity
        r[t] = (p_r[t] - lr - b[i]) / (2 * c[i]) - Pg_min[i]
        if r[t] > Rmax:
            r[t] = Rmax

        if r[t] < Rmin:
            r[t] = Rmin

        Nmin = 0
        Nmax = ramp_rate[i] / 60 * 10 - r[t]
        n[t] = (p_n[t] - lr - b[i]) / (2 * c[i]) - Pg_min[i] - r[t]
        if n[t] > Nmax:
            n[t] = Nmax

        if n[t] < Nmin:
            n[t] = Nmin

    if p_n[t] > p_r[t]:
        Nmin = 0
        Nmax = ramp_rate[i] / 60 * 10
        n[t] = (p_n[t] - lr - b[i]) / (2 * c[i]) - Pg_min[i]
        if n[t] > Nmax:
            n[t] = Nmax

        if n[t] < Nmin:
            n[t] = Nmin

        Rmin = 0
        Rmax = ramp_rate[i] / 60 * 10 - n[t]
        r[t] = (p_r[t] - lr - b[i]) / (2 * c[i]) - Pg_min[i] - n[t]
        if r[t] > Rmax:
            r[t] = Rmax

        if r[t] < Rmin:
            r[t] = Rmin

    Pmin = Pg_min[i]
    Pmax = Pg_max[i] - n[t] - r[t]
    p[t] = (p_g[t] - lr - b[i]) / (2 * c[i]) - r[t] - n[t]
    if p[t] > Pmax:
        p[t] = Pmax

    if p[t] < Pmin:
        p[t] = Pmin

    Cost = a[i] + b[i] * (p[t] + n[t] + r[t]) + c[i] * (p[t] + n[t] + r[t]) ** 2 - f_bl[i, t]
    L[t] = (-p_g[t] * (p[t] - P0[i, t]) + lg * p[t] - p_r[t] * r[t] + lr * r[t] - p_n[t] * n[t] + ln * n[t]) + Cost
    cost_on[t] = L[t]

    # Unit Off
    Noffmin = 0
    Noffmax = quick_start[i]
    n_off[t] = (p_n[t] - lr - b[i]) / (2 * c[i])
    if n_off[t] > Noffmax:
        n_off[t] = Noffmax

    if n_off[t] < Noffmin:
        n_off[t] = Noffmin

    if n_off[t] == 0:
        B = P0[i, t]
        cost_off[t] = -f_bl[i, t] + p_g[t] * B
    else:
        B = P0[i, t]
        Cost = a[i] + b[i] * (n_off[t]) + c[i] * (n_off[t]) ** 2 + p_g[t] * B
        cost_off[t] = - p_n[t] * n_off[t] + ln * n_off[t] - f_bl[i, t] + Cost

print('L: ', L)
print('r: ', r)
print('n: ', n)
print('p: ', p)
print('n_off: ', n_off)
print('startup: ', startup[i])
print('cost_on: ', cost_on)
print('cost_off: ', cost_off)

nodes = {
    0: (0, initialState[i]), 1: (1, 0), 2: (1, 1), 3: (2, 0), 4: (2, 1), 5: (3, 0),
    6: (3, 1), 7: (4, 0), 8: (4, 1)
}

if min(cost_off) <= 0 or min(cost_on) <= 0:
    bias = min(min(cost_off), min(cost_on)) - 1
else:
    bias = 0

max_cost = max(max(cost_on), max(cost_off))
scale = 5/(max_cost - bias)

print('bias: ', bias)


def cost_callback(from_node, to_node):
    _cost = 0
    t1 = to_node[0]
    t0 = from_node[0]

    if t1 - t0 == 1:  # must go one time unit forward
        if to_node[1] > 0:  # going to on state
            _cost = cost_on[t1]
            if from_node[1] <= 0:  # coming from off state
                _cost = _cost + startup[i]
        elif to_node[1] <= 0:  # going to off state
            _cost = cost_off[t1]
            if from_node[1] > 0:  # coming from on state
                _cost = _cost + shutdown[i]
        return (_cost - bias)*scale  # make sure costs are all greater than 0
    else:
        return np.inf


class AntColonyPBUC(AntColony):
    class Ant(AntColony.Ant):
        def _update_route(self, new):
            """
            add new node to self.route
            remove new node from self.possible_location
            called from _traverse() & __init__()
            """
            # Switching constraints (min off time, min on time)
            if len(self.route) == 0:  # if starting state
                if initialState[i]:  # if starting in on state
                    self.hours_on = initialHour[i]
                    self.hours_off = 0
                else:  # if starting in off state
                    self.hours_off = initialHour[i]
                    self.hours_on = 0
            else:
                states = [nodes[key][1] for key in self.route]
                last_hour = len(self.route) - 1
                count = 0
                this_state = nodes[new][1]
                last_state = states[last_hour]
                while this_state == last_state:
                    count += 1
                    if last_hour == 0:
                        break
                    this_hour = last_hour
                    last_hour -= 1
                    this_state = states[this_hour]
                    last_state = states[last_hour]

                if nodes[new][1]:  # new state is on
                    if nodes[self.route[-1]][1]:
                        self.hours_on += count
                        self.hours_off = 0
                    else:
                        self.hours_on = 1
                        self.hours_off = 0
                else:
                    if nodes[self.route[-1]][1] == 0:
                        self.hours_off += count
                        self.hours_on = 0
                    else:
                        self.hours_off = 1
                        self.hours_on = 0

            if (nodes[new][1] and self.hours_on > minOn[i]) or (~nodes[new][1] and self.hours_off > minOff[i]):
                self.can_switch = True
            else:
                self.can_switch = False

            self.route.append(new)
            self.possible_locations.remove(new)
            from copy import deepcopy
            check_locations = deepcopy(self.possible_locations)
            for key in check_locations:
                if (nodes[key][0] <= nodes[new][0]) or (not self.can_switch and nodes[key][1] != nodes[new][1] and nodes[key][0] == nodes[new][0] + 1):
                    self.possible_locations.remove(key)

        def get_distance_traveled(self):
            if self.tour_complete:
                return self.distance_traveled
            return None

        def _pick_path(self):
            """
            source: https://en.wikipedia.org/wiki/Ant_colony_optimization_algorithms#Edge_selection
            implements the path selection algorithm of ACO
            calculate the attractiveness of each possible transition from the current location
            then randomly choose a next path, based on its attractiveness
            """
            # on the first pass (no pheromones), then we can just choice() to find the next one
            if self.first_pass:
                import random
                return random.choice(self.possible_locations)

            attractiveness = dict()
            sum_total = 0.0
            # for each possible location, find its attractiveness
            # (it's (pheromone amount)*1/distance [tau*eta, from the algorithm])
            # sum all attractiveness amounts for calculating probability of each route in the next step
            for possible_next_location in self.possible_locations:
                # NOTE: do all calculations as float, otherwise we get integer division at times
                # for really hard to track down bugs
                pheromone_amount = float(self.pheromone_map[self.location][possible_next_location])
                distance = float(self.distance_callback(self.location, possible_next_location))

                # tau^alpha * eta^beta
                # pheromone_amount + 0.1 for cases when they are all zero
                attractiveness[possible_next_location] = pow(pheromone_amount + 0.1, self.alpha) * pow(1 / distance,
                                                                                                       self.beta)
                sum_total += attractiveness[possible_next_location]

            # it is possible to have small values for pheromone amount / distance, such that with
            # rounding errors this is equal to zero
            # rare, but handle when it happens
            if sum_total == 0.0:
                # increment all zero's, such that they are the smallest non-zero values supported by the system
                # source: http://stackoverflow.com/a/10426033/5343977
                def next_up(x):
                    import math
                    import struct
                    # NaNs and positive infinity map to themselves.
                    if math.isnan(x) or (math.isinf(x) and x > 0):
                        return x

                    # 0.0 and -0.0 both map to the smallest +ve float.
                    if x == 0.0:
                        x = 0.0

                    n = struct.unpack('<q', struct.pack('<d', x))[0]

                    if n >= 0:
                        n += 1
                    else:
                        n -= 1
                    return struct.unpack('<d', struct.pack('<q', n))[0]

                for key in attractiveness:
                    attractiveness[key] = next_up(attractiveness[key])
                sum_total = next_up(sum_total)

            # cumulative probability behavior, inspired by: http://stackoverflow.com/a/3679747/5343977
            # randomly choose the next path
            import random
            toss = random.random()
            cumulative = 0
            for possible_next_location in attractiveness:
                weight = (attractiveness[possible_next_location] / sum_total)
                if toss <= weight + cumulative:
                    return possible_next_location
                cumulative += weight


colony = AntColonyPBUC(nodes, cost_callback, ant_count=10, alpha=0.5, beta=1.2, iterations=100)  # alpha=0.5, beta=1.2
route = colony.mainloop()
print(route)
print(colony.shortest_distance/scale + bias * n_hours)
print(colony.shortest_distance)

t = [nodes[key][0] for key in route]
state = [nodes[key][1] for key in route]
plt.figure(1)
plt.plot(t, state, marker='o')
plt.figure(2)
plt.plot(colony.it_best)
plt.show()
