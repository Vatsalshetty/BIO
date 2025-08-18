import random
import numpy as np
from sklearn.model_selection import cross_val_score

def initialize_population(pop_size, num_features):
    # Each individual is a bitmask of length num_features
    return [np.random.randint(0, 2, num_features).tolist() for _ in range(pop_size)]

def fitness(individual, X, y, estimator, cv=5, scorer=None):
    # Select features according to the bitmask
    if sum(individual) == 0:
        return 0  # Avoid empty feature set
    X_selected = X[:, [i for i, bit in enumerate(individual) if bit == 1]]
    # Use cross-validation score as fitness
    scores = cross_val_score(estimator, X_selected, y, cv=cv, scoring=scorer)
    return scores.mean()

def calculate_fitness(population, X, y, estimator, cv=5, scorer=None):
    return [fitness(ind, X, y, estimator, cv, scorer) for ind in population]

def select_mating_pool(population, fitness_scores, num_parents):
    total_fitness = sum(fitness_scores)
    if total_fitness == 0:
        return random.sample(population, num_parents)
    selection_probs = [score / total_fitness for score in fitness_scores]
    parents = random.choices(population, weights=selection_probs, k=num_parents)
    return parents

def crossover(parents, offspring_size):
    offspring = []
    num_features = len(parents[0])
    for _ in range(offspring_size):
        parent1 = random.choice(parents)
        parent2 = random.choice(parents)
        crossover_point = random.randint(1, num_features - 1)
        child = parent1[:crossover_point] + parent2[crossover_point:]
        offspring.append(child)
    return offspring

def mutate(population, mutation_rate=0.1):
    mutated_pop = []
    num_features = len(population[0])
    for individual in population:
        mutated = individual[:]
        for i in range(num_features):
            if random.random() < mutation_rate:
                mutated[i] = 1 - mutated[i]  # Flip bit
        mutated_pop.append(mutated)
    return mutated_pop

def is_converged(population, prev_population):
    return population == prev_population

def genetic_feature_selection(X, y, estimator, pop_size=20, max_generations=50, num_parents=8, mutation_rate=0.1, patience=10, cv=5, scorer=None):
    num_features = X.shape[1]
    population = initialize_population(pop_size, num_features)
    fitness_scores = calculate_fitness(population, X, y, estimator, cv, scorer)
    best_idx = np.argmax(fitness_scores)
    best = population[best_idx]
    best_fitness = fitness_scores[best_idx]
    unchanged_generations = 0

    for gen in range(max_generations):
        parents = select_mating_pool(population, fitness_scores, num_parents)
        offspring = crossover(parents, pop_size - num_parents)
        offspring = mutate(offspring, mutation_rate)
        prev_population = population
        population = parents + offspring
        fitness_scores = calculate_fitness(population, X, y, estimator, cv, scorer)
        current_best_idx = np.argmax(fitness_scores)
        current_best = population[current_best_idx]

        if fitness_scores[current_best_idx] > best_fitness:
            best = current_best
            best_fitness = fitness_scores[current_best_idx]
            unchanged_generations = 0
        elif is_converged(population, prev_population):
            unchanged_generations += 1
        else:
            unchanged_generations = 0

        print(f"Generation {gen+1}: Best fitness={best_fitness}, Selected features={sum(best)}")
        if unchanged_generations >= patience:
            print(f"Converged after {gen+1} generations.")
            break

    return best, best_fitness

# Example usage:
if __name__ == "__main__":
    # Demo with sklearn's breast cancer dataset
    from sklearn.datasets import load_breast_cancer
    from sklearn.ensemble import RandomForestClassifier

    data = load_breast_cancer()
    X = data.data
    y = data.target

    estimator = RandomForestClassifier(n_estimators=50, random_state=42)
    best_mask, best_score = genetic_feature_selection(X, y, estimator, scorer='accuracy')
    selected_features = [i for i, bit in enumerate(best_mask) if bit == 1]
    print(f"Optimal features: {selected_features}")
    print(f"Best accuracy: {best_score:.4f}")
