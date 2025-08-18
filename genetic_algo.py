import random

def initialize_population(size, min_x=0, max_x=31):
    return [random.randint(min_x, max_x) for _ in range(size)]

def fitness(x):
    return x ** 2

def calculate_fitness(population):
    return [fitness(individual) for individual in population]

def select_mating_pool(population, fitness_scores, num_parents):
    total_fitness = sum(fitness_scores)
    if total_fitness == 0:
        return random.sample(population, num_parents)
    selection_probs = [score / total_fitness for score in fitness_scores]
    parents = random.choices(population, weights=selection_probs, k=num_parents)
    return parents

def crossover(parents, offspring_size):
    offspring = []
    for _ in range(offspring_size):
        parent1 = random.choice(parents)
        parent2 = random.choice(parents)
        crossover_point = random.randint(1, 4)
        mask = (1 << crossover_point) - 1
        child = (parent1 & ~mask) | (parent2 & mask)
        offspring.append(child)
    return offspring

def mutate(population, mutation_rate=0.1):
    mutated_pop = []
    for individual in population:
        if random.random() < mutation_rate:
            bit_to_flip = 1 << random.randint(0, 4)
            individual ^= bit_to_flip
        mutated_pop.append(individual)
    return mutated_pop

def is_converged(population, prev_population):
    return sorted(population) == sorted(prev_population)

def genetic_algorithm(pop_size=10, max_generations=100, num_parents=4, mutation_rate=0.1, patience=10):
    population = initialize_population(pop_size)
    best = max(population, key=fitness)
    best_fitness = fitness(best)
    unchanged_generations = 0

    for gen in range(max_generations):
        fitness_scores = calculate_fitness(population)
        parents = select_mating_pool(population, fitness_scores, num_parents)
        offspring = crossover(parents, pop_size - num_parents)
        offspring = mutate(offspring, mutation_rate)
        prev_population = population
        population = parents + offspring

        current_best = max(population, key=fitness)
        if fitness(current_best) > best_fitness:
            best = current_best
            best_fitness = fitness(current_best)
            unchanged_generations = 0
        elif is_converged(population, prev_population):
            unchanged_generations += 1
        else:
            unchanged_generations = 0

        print(f"Generation {gen+1}: Best x={best}, Fitness={best_fitness}")
        if unchanged_generations >= patience:
            print(f"Converged after {gen+1} generations.")
            break

    return best, best_fitness

if __name__ == "__main__":
    best, best_fitness = genetic_algorithm()
    print(f"Optimal solution found: x={best}, Fitness={best_fitness}")
