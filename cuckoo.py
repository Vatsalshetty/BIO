import numpy as np
import random
import math
from itertools import permutations # Useful for smaller tests, but not scalable

# --- 0. TSP Problem Definition ---
# Let's define some example cities with (x, y) coordinates
# In a real scenario, you'd load this from a file or generate it.
CITIES_COORDS = {
    0: (0, 0),
    1: (1, 3),
    2: (4, 1),
    3: (2, 5),
    4: (5, 2),
    5: (3, 0)
}
NUM_CITIES = len(CITIES_COORDS)
CITY_INDICES = list(range(NUM_CITIES))

# Calculate a distance matrix between cities
DISTANCE_MATRIX = np.zeros((NUM_CITIES, NUM_CITIES))
for i in range(NUM_CITIES):
    for j in range(NUM_CITIES):
        if i == j:
            DISTANCE_MATRIX[i, j] = 0
        else:
            x1, y1 = CITIES_COORDS[i]
            x2, y2 = CITIES_COORDS[j]
            DISTANCE_MATRIX[i, j] = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

# --- 1. Objective Function for TSP ---
# This function calculates the total distance of a given tour (permutation of cities)
def calculate_tour_distance(tour):
    distance = 0
    for i in range(NUM_CITIES - 1):
        distance += DISTANCE_MATRIX[tour[i], tour[i+1]]
    distance += DISTANCE_MATRIX[tour[-1], tour[0]] # Return to the starting city
    return distance

# --- 2. TSP-Specific Mutation Operators (Mimicking Lévy Flight) ---
# These are crucial replacements for the simple additive Lévy flight for continuous problems.

def apply_2_opt_swap(tour):
    """
    Performs a 2-opt swap mutation (local exploitation).
    Picks two random points in the tour and reverses the segment between them.
    """
    n = len(tour)
    new_tour = list(tour) # Create a mutable copy
    
    # Pick two distinct points
    i, j = sorted(random.sample(range(n), 2))
    
    # Reverse the segment between i and j
    new_tour[i:j+1] = new_tour[i:j+1][::-1]
    return new_tour

def apply_scramble_mutation(tour):
    """
    Performs a scramble mutation (global exploration).
    Picks a random segment and scrambles the order of cities within it.
    """
    n = len(tour)
    new_tour = list(tour) # Create a mutable copy

    # Pick two distinct points to define a segment
    i, j = sorted(random.sample(range(n), 2))

    # Extract the segment, scramble it, and put it back
    segment = new_tour[i:j+1]
    random.shuffle(segment)
    new_tour[i:j+1] = segment
    return new_tour

# Combine mutations with a probability to simulate Lévy flight
def generate_new_solution_tsp(current_best_tour):
    """
    Generates a new TSP solution from an existing one,
    mimicking Lévy flight with a mix of small and large mutations.
    """
    if random.random() < 0.8: # High probability of local search (exploitation)
        return apply_2_opt_swap(current_best_tour)
    else: # Low probability of global search (exploration)
        return apply_scramble_mutation(current_best_tour)

# --- Cuckoo Search Algorithm for TSP ---
def cuckoo_search_tsp(num_nests, num_iterations, pa):
    # STEP 1: Initialize the Nests
    # Each nest is a random permutation of cities (a random tour)
    nests = []
    for _ in range(num_nests):
        tour = list(CITY_INDICES) # Start with ordered cities
        random.shuffle(tour)      # Shuffle to get a random permutation
        nests.append(tour)

    fitness = np.array([calculate_tour_distance(tour) for tour in nests])

    # Keep track of the best solution found so far
    best_nest_idx = np.argmin(fitness)
    best_tour = list(nests[best_nest_idx]) # Store as mutable list
    best_fitness = fitness[best_nest_idx]

    print(f"Initial best tour: {best_tour} with distance: {best_fitness:.2f}")

    # --- Main Algorithm Loop ---
    for iteration in range(num_iterations):
        # STEP 2: Generate New Cuckoos (TSP-specific Mutation)
        # Pick a cuckoo (a random nest) and generate a new solution
        i = random.randint(0, num_nests - 1)
        
        # Generate new tour using our TSP-specific mutation
        new_tour = generate_new_solution_tsp(nests[i])
        new_fitness = calculate_tour_distance(new_tour)

        # Compare the new cuckoo with another random nest
        j = random.randint(0, num_nests - 1)
        if new_fitness < fitness[j]:
            nests[j] = new_tour
            fitness[j] = new_fitness

        # STEP 3: Abandon Worst Nests
        # A fraction (pa) of the worst nests are abandoned and replaced with new random ones.
        num_to_abandon = int(pa * num_nests)
        if num_to_abandon > 0: # Ensure we don't try to sort empty list if pa is too small
            worst_indices = np.argsort(fitness)[-num_to_abandon:] # Find indices of worst nests
            
            for idx in worst_indices:
                new_random_tour = list(CITY_INDICES)
                random.shuffle(new_random_tour)
                nests[idx] = new_random_tour
                fitness[idx] = calculate_tour_distance(nests[idx])
            
        # Update the overall best solution found
        current_best_idx = np.argmin(fitness)
        if fitness[current_best_idx] < best_fitness:
            best_tour = list(nests[current_best_idx])
            best_fitness = fitness[current_best_idx]
            
        # Optional: Print progress
        if (iteration + 1) % 10 == 0 or iteration == 0:
            print(f"Iteration {iteration+1}: Best distance = {best_fitness:.2f}, Tour: {best_tour}")

    return best_tour, best_fitness

# --- Run the Algorithm ---
if __name__ == "__main__":
    # Parameters for TSP Cuckoo Search
    NUM_NESTS_TSP = 30          # Number of tours in the population
    NUM_ITERATIONS_TSP = 500    # Number of cycles
    P_ABANDON_TSP = 0.2         # Probability of a nest being abandoned (pa)

    print(f"Number of cities: {NUM_CITIES}")
    print("City Coordinates:")
    for city, coord in CITIES_COORDS.items():
        print(f"  City {city}: {coord}")
    print("\nStarting Cuckoo Search for TSP...")

    final_tour, final_distance = cuckoo_search_tsp(
        num_nests=NUM_NESTS_TSP,
        num_iterations=NUM_ITERATIONS_TSP,
        pa=P_ABANDON_TSP
    )
    
    print("\nCuckoo Search for TSP finished.")
    print(f"Final best tour: {final_tour}")
    print(f"Final best distance: {final_distance:.2f}")

    # For verification on small numbers of cities (e.g., up to 10-12 cities)
    # you can compare with the true brute-force optimal solution.
    # For NUM_CITIES = 6, the number of permutations is 5! = 120, which is manageable.
    if NUM_CITIES <= 10:
        print("\n--- Verifying with Brute Force (for small N) ---")
        min_brute_force_distance = float('inf')
        optimal_brute_force_tour = None
        
        # We need to fix a starting city for permutations to avoid duplicates.
        # Let's assume starting from city 0, then permute the rest.
        other_cities = list(range(1, NUM_CITIES)) 
        for p in permutations(other_cities):
            current_tour = [0] + list(p)
            current_distance = calculate_tour_distance(current_tour)
            if current_distance < min_brute_force_distance:
                min_brute_force_distance = current_distance
                optimal_brute_force_tour = current_tour
        
        print(f"Brute Force Optimal Tour: {optimal_brute_force_tour}")
        print(f"Brute Force Optimal Distance: {min_brute_force_distance:.2f}")
        print(f"Cuckoo Search found difference: {final_distance - min_brute_force_distance:.2f}")
