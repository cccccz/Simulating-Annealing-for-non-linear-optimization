import random
import numpy as np
from matplotlib import pyplot as plt


def read_cities(filename):
    cities = []
    with open(f'data/{filename}.data', 'r') as handle:
        lines = handle.readlines()
        for line in lines:
            x, y = map(float, line.split())
            cities.append([x, y])
    return cities


def distance(city1, city2):
    return np.sqrt((city1[0] - city2[0]) ** 2 + (city1[1] - city2[1]) ** 2)


def generate_distance_matrix(cities):
    #Generate a distance matrix from a list of cities
    num_cities = len(cities)
    distances = np.zeros((num_cities, num_cities))
    for i in range(num_cities):
        for j in range(num_cities):
            distances[i, j] = distance(cities[i], cities[j])
    return distances


def cost_function(path, distances):
    cost = 0
    for i in range(len(path) - 1):
        cost += distances[path[i], path[i + 1]]
    cost += distances[path[-1], path[0]]
    return cost


#a random initial solution
def generate_initial_solution(num_cities):
    path = list(range(num_cities))
    random.shuffle(path)
    return path


#SA algorithm
def simulated_annealing(distances, temperature=20000, cooling_rate=0.0001):
    # Set up
    current_solution = generate_initial_solution(len(distances))
    best_solution = current_solution.copy()
    current_cost = cost_function(current_solution, distances)
    print("initial cost is:", current_cost)
    best_cost = current_cost

    # keep track of costs for the plot
    costs = []
    # loop
    while temperature > 1:
        # Generate a new neighbor solution
        neighbor_solution = current_solution.copy()
        i, j = sorted(random.sample(range(len(current_solution)), 2))
        neighbor_solution[i:j + 1] = reversed(neighbor_solution[i:j + 1])
        neighbor_cost = cost_function(neighbor_solution, distances)

        #accept or not
        delta_cost = neighbor_cost - current_cost
        acceptance_probability = np.exp(-delta_cost / temperature)
        if acceptance_probability > random.random():
            current_solution = neighbor_solution
            current_cost = neighbor_cost

        # Update the best solution
        if current_cost < best_cost:
            best_solution = current_solution.copy()
            best_cost = current_cost

        # Lower the temperature
        temperature *= 1 - cooling_rate

        costs.append(best_cost)
    print("temp:",temperature, "cooling rate:", cooling_rate, "cost:",best_cost)
    return best_solution, best_cost, costs


def visualize_cities_and_path(cities, path):
    x = [point[0] for point in cities]
    y = [point[1] for point in cities]

    path2 = path + [path[0]]
    path_x = [x[path2[i]] for i in range(len(path2))]
    path_y = [y[path2[i]] for i in range(len(path2))]

    #line plot of the path over the scatter plot of the cities
    plt.plot(path_x, path_y, 'r-')
    plt.scatter(x, y)
    plt.title("Cities and Paths")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

def visualize_cities(cities):
    x = [point[0] for point in cities]
    y = [point[1] for point in cities]

    plt.scatter(x, y)
    plt.title("Cities and Paths")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


def plot_costs(costs):
    plt.plot(costs)
    plt.ylabel('Cost')
    plt.xlabel('Iteration')
    plt.title('Cost vs Iteration')
    plt.show()

def plot_multiple_costs(parameters,distances):
    for temperature, cooling_rate in parameters:
        _, _, costs = simulated_annealing(distances, temperature, cooling_rate)
        plt.plot(costs, label=f'Temp={temperature}, CR={cooling_rate}')

    plt.ylabel('Cost')
    plt.xlabel('Iteration')
    plt.title('Cost vs Iteration for different parameters')
    plt.legend()
    plt.show()

parameters0 = [(20000,1e-4),(20000,1e-5),(20000,1e-3),(20000,1e-2)]
parameters1 = [(20000,1e-4),(10000,1e-4),(5000,1e-4),(1000,1e-4)]
parameters2 = [(20000,1e-4),(20000,1e-5),(20000,1e-3),(20000,1e-2),
               (10000,1e-4),(10000,1e-5),(10000,1e-3),(10000,1e-2),
               (5000,1e-4),(5000,1e-5),(5000,1e-3),(5000,1e-2)]
cities = read_cities('cities')
distances = generate_distance_matrix(cities)
random.seed(42)
path = generate_initial_solution(len(cities))
# path = [0,1,2,3]
visualize_cities_and_path(cities,path)
best_tour, best_cost, costs = simulated_annealing(distances)
print("The best cost is:",best_cost)
print("the best path is:",best_tour)
visualize_cities_and_path(cities, best_tour)
#plot_multiple_costs(parameters0,distances)
#plot_multiple_costs(parameters1,distances)
plot_multiple_costs(parameters2,distances)

