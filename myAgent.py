import matplotlib.pyplot as plt
import numpy as np
import random
import defaults

playerName = "myAgent"
nPercepts = 75  # This is the number of percepts
nActions = 7  # This is the number of actions
nChromosome = 9  # Length of a chromosome
tournament_winners = 5
game_count = 0  # Keep track of the game count
generation_fitness = np.zeros(defaults.game_params['nGames'])
temp_array = np.zeros(100)
create_chart = False


class MyCreature:

    def __init__(self):
        self.fitness = 0
        self.chromosome = np.zeros(nChromosome)
        for i in range(nChromosome):
            self.chromosome[i] = random.randint(0, 100)

    # Receives a tensor or percepts and maps the percepts plus chromosomes to actions.
    def AgentFunction(self, percepts):

        creature_map = percepts[:, :, 0]  # 5x5 map with information about creatures and their size
        food_map = percepts[:, :, 1]  # 5x5 map with information about strawberries
        wall_map = percepts[:, :, 2]  # 5x5 map with information about walls

        small_enemy = False
        large_enemy = False
        food = False
        food_direction = 0
        enemy_direction = 0

        # The three personalities types, actions depend on the strength of each personality type
        hunter = self.chromosome[0] + self.chromosome[1] + self.chromosome[2]
        hungary = self.chromosome[3] + self.chromosome[4] + self.chromosome[5]
        coward = self.chromosome[6] + self.chromosome[7] + self.chromosome[8]

        my_size = np.abs(creature_map[2, 2])  # My creatures size, used for comparison to other creatures
        actions = np.zeros(nActions)

        # These for loops and if statements check for any food and enemy creatures in the 5 by 5 percepts
        # if any are true then the corresponding variables are set to True, there are two different variables
        # for enemy (large and small), if the creature is equal to my creatures size then are set under the large
        # variable.
        for i in range(0, 5):
            for j in range(0, 5):
                if food_map[j, i] == 0 and creature_map[j, i] == 0:
                    continue
                if food_map[j, i] == 1:
                    food = True
                    food_direction = j + i
                elif np.abs(creature_map[j, i]) > 0 and creature_map[j, i] != creature_map[2, 2]:
                    if np.abs(creature_map[j, i]) < my_size:
                        small_enemy = True
                        enemy_direction = i + j
                    elif np.abs(creature_map[j, i]) == my_size:
                        continue
                    else:
                        large_enemy = True
                        enemy_direction = i + j

        # These if blocks control the creatures directional actions (actions[0 - 4]) e.g., if there is a small
        # creature and food then it depends on the creatures personality on what happens. If there are no
        # large, small creatures or food then no actions are set.
        if small_enemy and food:
            if hungary >= hunter and hungary > coward:
                if food_map[2, 2] == 1:
                    actions[5] += self.chromosome[3]
                elif food_direction > 4:
                    actions[2] += self.chromosome[4]
                    actions[3] += self.chromosome[5]
                else:
                    actions[0] += self.chromosome[3]
                    actions[1] += self.chromosome[4]
            else:
                if enemy_direction > 4:
                    actions[2] += self.chromosome[0]
                    actions[3] += self.chromosome[1]
                else:
                    actions[0] += self.chromosome[2]
                    actions[1] += self.chromosome[0]
        elif large_enemy and food:
            if hungary >= coward:
                if food_map[2, 2] == 1:
                    actions[5] += self.chromosome[3]
                elif food_direction > 4:
                    actions[2] += self.chromosome[4]
                    actions[3] += self.chromosome[5]
                else:
                    actions[0] += self.chromosome[3]
                    actions[1] += self.chromosome[4]
            else:
                if enemy_direction < 4:
                    actions[2] += self.chromosome[0]
                    actions[3] += self.chromosome[1]
                else:
                    actions[0] += self.chromosome[2]
                    actions[1] += self.chromosome[0]
        else:
            if food:
                if food_map[2, 2] == 1:
                    actions[5] += self.chromosome[3]
                elif food_direction > 4:
                    actions[2] += self.chromosome[4]
                    actions[3] += self.chromosome[3]
                else:
                    actions[0] += self.chromosome[3]
                    actions[1] += self.chromosome[5]
            elif small_enemy:
                if enemy_direction > 4:
                    actions[2] += self.chromosome[2]
                    actions[3] += self.chromosome[1]
                else:
                    actions[0] += self.chromosome[0]
                    actions[1] += self.chromosome[1]
            elif large_enemy:
                if enemy_direction < 4:
                    actions[2] += self.chromosome[6]
                    actions[3] += self.chromosome[7]
                else:
                    actions[0] += self.chromosome[7]
                    actions[1] += self.chromosome[6]

        # This code is designed to prevent wasted moves (not trying to move onto a wall or a friendly creature) For
        # example if there is a wall above the creature (wall_map[2, 1]) and they have an action score to move
        # up, then append the corresponding chromosome for the action score for moving down
        # (away from the wall or friendly creature).
        if np.abs(wall_map[2, 1]) == 1 or np.abs(creature_map[2, 1] and actions[0] != 0) < 0:
            actions[0] = 0
        if np.abs(wall_map[1, 2]) == 1 or np.abs(creature_map[1, 2] and actions[1] != 0) < 0:
            actions[1] = 0
        if np.abs(wall_map[2, 3]) == 1 or np.abs(creature_map[2, 3] and actions[2] != 0) < 0:
            actions[2] = 0
        if np.abs(wall_map[3, 2]) == 1 or np.abs(creature_map[3, 2] and actions[3] != 0) < 0:
            actions[3] = 0

        # Do nothing
        actions[4] = self.chromosome[6]

        # Move in random direction
        actions[6] = self.chromosome[7]

        return actions


# Implements the fitness function, parent selection, elitism, the uniform cross-over method and mutations.
def newGeneration(old_population):
    global game_count
    length_old_population = len(old_population)
    overall_fitness = np.zeros(length_old_population)

    # My fitness function, a creature fitness equals the sum of the creatures turn, size, strawberries eaten
    # and enemy eaten (with different emphasis on the different attributes). For example I considered the creature
    # living through the game the most important so if a creature lived through a game then their fitness score was
    # doubled.
    for n, creature in enumerate(old_population):
        creature.fitness = ((creature.turn / 10) + (
                (creature.size + creature.strawb_eats + creature.enemy_eats) * 2) * (int(creature.alive) * 2)) / 100
        overall_fitness[n] = creature.fitness

    # Sort the old_population in order of fitness
    old_population.sort(key=lambda x: x.fitness, reverse=True)

    new_population = list()

    # The below code selects n number of the fittest creatures (n = tournament_winners) and picks two random creatures
    # from the tournament winners, the two picked creatures are used as parent_one and parent_two while the remaining
    # tournament winners were added to the new population (elitism).
    selection = random.sample(range(tournament_winners), k=2)
    pick_one = selection[0]
    pick_two = selection[1]

    parent_one = old_population[pick_one]
    parent_two = old_population[pick_two]

    for i in range(tournament_winners):
        if i != pick_one or i != parent_two:
            new_population.append(old_population[i])

    # For the crossover I used the uniform cross-over method. I added a 5% chance of a random mutation.
    for n in range(length_old_population - tournament_winners):
        new_creature = MyCreature()
        for i in range(nChromosome):
            if i % 2 == 0:
                new_creature.chromosome[i] = parent_one.chromosome[i]
            else:
                new_creature.chromosome[i] = parent_two.chromosome[i]
            # Five percent chance of a mutation
            if random.randint(0, 99) <= 5:
                new_creature.chromosome[random.randint(0, nChromosome - 1)] = random.randint(0, 99)

        # Add the new agent to the new population
        new_population.append(new_creature)

    # At the end you need to compute average fitness and return it along with your new population
    avg_fitness = np.mean(overall_fitness)
    generation_fitness[game_count] = avg_fitness
    game_count += 1

    # To calculate and display the average fitness chart.
    if game_count == defaults.game_params['nGames'] and create_chart:
        temp = 0
        count = 0
        for i in range(defaults.game_params['nGames'] + 1):
            temp += generation_fitness[i - 1]
            if i % 5 == 0 and i != 0:
                temp_array[count] = temp / 5
                count += 1
                temp = 0
        plt.plot(temp_array)
        plt.xlabel("Generation")
        plt.ylabel("Average Fitness")
        plt.title("Change in average Fitness")
        plt.savefig("fitness_chart.png")

    return new_population, avg_fitness
