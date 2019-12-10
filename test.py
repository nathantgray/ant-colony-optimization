from antcolony import AntColony
import matplotlib.pyplot as plt
import numpy as np

test_nodes = {
    0: (0, 7), 1: (3, 9), 2: (12, 4), 3: (14, 11), 4: (8, 11),
    5: (15, 6), 6: (6, 15), 7: (15, 9), 8: (12, 10), 9: (10, 7)
}


def distance_callback(from_node, to_node):
    dx = to_node[0] - from_node[0]
    dy = to_node[1] - from_node[1]
    return (dx**2 + dy**2)**(1/2)


colony = AntColony(test_nodes, distance_callback, ant_count=1, alpha=0.5, beta=1.2, iterations=5)  # alpha=0.5, beta=1.2
route = colony.mainloop()
print(route)
print(colony.shortest_distance)

x = [test_nodes[key][0] for key in route]
y = [test_nodes[key][1] for key in route]
plt.figure(1)
plt.plot(x, y, marker='o')
plt.figure(2)
plt.plot(colony.it_best)
plt.show()