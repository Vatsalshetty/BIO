import random
import numpy as np 


def fitness_function(position):
    return np.sum(position**2)


def particle_swarm_optimization(dimensions, num_particles, max_iterations):
    w = 0.5   
    c1 = 0.8  # Personal component coefficient
    c2 = 0.9  # Social component coefficient

    # --- Step 1: Initialize Swarm ---
    swarm = []
    for _ in range(num_particles):
        position = np.random.uniform(-10, 10, size=dimensions)
        velocity = np.random.uniform(-1, 1, size=dimensions)
        pbest_position = position.copy()
        pbest_fitness = fitness_function(position)
        swarm.append({'position': position, 'velocity': velocity,
                      'pbest_position': pbest_position, 'pbest_fitness': pbest_fitness})

    # Initialize global best
    gbest_position = np.zeros(dimensions)
    gbest_fitness = float('inf')

    # --- Step 2: Start Main Loop ---
    for i in range(max_iterations):
        # --- Update Bests ---
        for p in swarm:
            fitness = fitness_function(p['position'])
            # Step 2b: Update Personal Best
            if fitness < p['pbest_fitness']:
                p['pbest_fitness'] = fitness
                p['pbest_position'] = p['position'].copy()
            # Step 2c: Update Global Best
            if fitness < gbest_fitness:
                gbest_fitness = fitness
                gbest_position = p['position'].copy()

        # --- Step 3: Update Movement ---
        for p in swarm:
            rand1 = random.random()
            rand2 = random.random()

            # Step 3a: Update Velocity
            inertia_term = w * p['velocity']
            personal_term = c1 * rand1 * (p['pbest_position'] - p['position'])
            social_term = c2 * rand2 * (gbest_position - p['position'])
            p['velocity'] = inertia_term + personal_term + social_term

            # Step 3b: Update Position
            p['position'] = p['position'] + p['velocity']

    # --- Step 5: Output ---
    print(f"SOLUTION FOUND:")
    print(f"  Position: {gbest_position}")
    print(f"  Fitness: {gbest_fitness}")
    return gbest_position, gbest_fitness

# --- Run the algorithm ---
particle_swarm_optimization(dimensions=2, num_particles=300, max_iterations=10)