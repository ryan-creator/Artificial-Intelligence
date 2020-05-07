import matplotlib.pyplot as plt
import numpy as np
import random
import defaults

playerName = "myAgent"
nPercepts = 75      # This is the number of percepts
nActions = 7        # This is the number of actions
nChromosome = 8     # Length of a chromosome
game_count = 0      # Keep track of the game count
fitness_list = []   # Store the overall_fitness for each game to plot on a char


class MyCreature:

    def __init__(self):
        self.fitness = 0
        self.chromosome = np.zeros(nChromosome)
        for i in range(7):
            self.chromosome[i] = random.randint(0, 100)

    # Receives a tensor or percepts and maps the percepts plus chromosomes to actions.
    def AgentFunction(self, percepts):

        creature_map = percepts[:, :, 0]    # 5x5 map with information about creatures and their size
        food_map = percepts[:, :, 1]        # 5x5 map with information about strawberries
        wall_map = percepts[:, :, 2]        # 5x5 map with information about walls

        my_size = np.abs(creature_map[2, 2])    # My creatures size, used for comparison to other creatures
        actions = np.zeros(nActions)

        # This wee if statement sets variables to particular values depending on the stage of the game,
        # This is done so the creatures can have different goals at different stages i.e. eat at the b
        # beginning to get bigger and then attack later in the game when bigger to win.
        if game_count > 50:
            game_stage_early = 50
            game_stage_late = 0
        else:
            game_stage_late = 50
            game_stage_early = 0

        # I choose to give the creatures personalities variables so their behaviour further differs, their
        # personality (aggressive or cautious) is determined by there 8th chromosome, an odd chromosome
        # equals cautious and even equals aggressive.
        personality_cautious = 0
        personality_aggressive = 0
        creature_character = self.chromosome[7]
        if creature_character % 2 == 0:
            personality_cautious = creature_character
        else:
            personality_aggressive = creature_character

        # This block of code guides creatures into making a decision about whether to fight or run when an
        # enemy creatures gets within the 5 by 5 map. If the creature is smaller than my creature then go
        # towards it (chase) otherwise move away (run). To understand where creatures are on the map, I
        # split the map into the sum(i, j) and either greater or less and 4, this split the map diagonally
        # (from the bottom left to the top right), any number less than 4 was in the top left section and
        # numbers bigger than 4 where in the bottom right section (finding the food also used the same mechanism).
        for i in range(0, 5):
            for j in range(0, 5):
                if creature_map[j, i] == 0:
                    continue
                if np.abs(creature_map[j, i]) > 0 and creature_map[j, i] != creature_map[2, 2]:
                    if np.abs(creature_map[j, i]) <= my_size:
                        # Chase & fight
                        if i + j >= 4:
                            actions[2] += self.chromosome[2] + personality_aggressive + game_stage_late
                            actions[3] += self.chromosome[3] + personality_aggressive + game_stage_late
                        else:
                            actions[0] += self.chromosome[0] + personality_aggressive + game_stage_late
                            actions[1] += self.chromosome[1] + personality_aggressive + game_stage_late
                    else:
                        # Run away
                        if i + j <= 4:
                            actions[2] += self.chromosome[2] + personality_cautious + game_stage_early
                            actions[3] += self.chromosome[3] + personality_cautious + game_stage_early
                        else:
                            actions[0] += self.chromosome[0] + personality_cautious + game_stage_early
                            actions[1] += self.chromosome[1] + personality_cautious + game_stage_early

        # This code used the same mechanisms explained for the fight run code, the creatures are guided to making a
        # decision into the section that the food is in. If the creature is on top of food then i have multiplied the
        # action score by 2 to increase the chances of the creature eating the food - but its not guaranteed e.g. if
        # the chromosome is very small then multiplying it by 2 wouldn't increase its chances by much.
        for i in range(0, 5):
            for j in range(0, 5):
                if food_map[j, i] == 0:
                    continue
                if food_map[j, i] == 1:
                    if food_map[2, 2] == 1:
                        actions[5] += (self.chromosome[5] + personality_cautious + game_stage_early) * 2
                    elif i + j <= 4:
                        actions[0] += self.chromosome[0] + personality_cautious + game_stage_early
                        actions[1] += self.chromosome[1] + personality_cautious + game_stage_early
                    else:
                        actions[2] += self.chromosome[2] + personality_cautious + game_stage_early
                        actions[3] += self.chromosome[3] + personality_cautious + game_stage_early

        # This code is designed to prevent wasted moves (not trying to move onto a wall or a friendly creature).
        if np.abs(wall_map[2, 1]) == 1 or np.abs(creature_map[2, 1]) < 0: actions[0] -= self.chromosome[0] / 2
        if np.abs(wall_map[1, 2]) == 1 or np.abs(creature_map[1, 2]) < 0: actions[1] -= self.chromosome[1] / 2
        if np.abs(wall_map[2, 3]) == 1 or np.abs(creature_map[2, 3]) < 0: actions[2] -= self.chromosome[2] / 2
        if np.abs(wall_map[3, 2]) == 1 or np.abs(creature_map[3, 2]) < 0: actions[3] -= self.chromosome[3] / 2

        # From watching replays I noticed that the creatures doing nothing looked extremely unhelpful so I wanted
        # to decrease the chance of this action being the action that's run, hence i divide the action score by 3.
        actions[4] = self.chromosome[4] / 3

        # Move in random direction
        actions[6] = self.chromosome[6]

        return actions

# Creates a new generation by picking two fittest parents through a tournament and creating a new population with
# their chromosomes.
def newGeneration(old_population):
    global game_count
    N = len(old_population)
    overall_fitness = np.zeros(N)

    # My fitness function, a creature fitness equals a summary of the creatures turn, size, strawberries eaten
    # and enemy eaten, if the creature remained alive their fitness score is multiplied by 0.5 to guarantee that
    # the parents were still alive at the end of the game, i.e. I considered being alive the most important attribute.
    for n, creature in enumerate(old_population):
        creature.fitness = ((creature.turn / 2) + creature.size + (
                    creature.strawb_eats + creature.enemy_eats * 2)) / 100
        if creature.alive:
            creature.fitness = creature.fitness * 0.5
        overall_fitness[n] = creature.fitness

    new_population = list()

    # The below code selects the two fittest creatures to become the parent one and two, and the third and forth
    # creatures to add straight to the new population (elitism). I did this to allow more diversity in the new
    # population.
    parent_one = creature
    parent_two = creature
    elitism_one = parent_two
    elitism_two = elitism_one
    for n, creature in enumerate(old_population):
        if creature.fitness > parent_one.fitness:
            elitism_two = elitism_one
            elitism_one = parent_two
            parent_two = parent_one
            parent_one = creature

    new_population.append(elitism_one)
    new_population.append(elitism_two)

    # For the crossover I used the uniform cross-over method. I noticed with the uniform cross-over (and the other
    # cross over methods I experimented with), that after a few generations there was very little diversity in the
    # populations chromosomes. So added a 5% chance of a random mutation, and the uniform cross-over will change
    # depending on the game, so every other game the parents are switched instead of the chromosome[0] coming
    # from parent_one, it will then come from parent_two. I found this still kept the attributes of the fittest
    # creatures but allowed more diversity in the population.
    for n in range(N - 2):
        new_creature = MyCreature()
        for i in range(nChromosome):
            if game_count % 2 == 0:
                if i % 2 == 0:
                    new_creature.chromosome[i] = parent_one.chromosome[i]
                else:
                    new_creature.chromosome[i] = parent_two.chromosome[i]
                # Five percent chance of a mutation
                if random.randint(0, 99) < 5:
                    new_creature.chromosome[random.randint(0, 6)] = random.randint(0, 99)
            else:
                if i % 2 == 1:
                    new_creature.chromosome[i] = parent_two.chromosome[i]
                else:
                    new_creature.chromosome[i] = parent_one.chromosome[i]
                # Five percent chance of a mutation
                if random.randint(0, 99) < 5:
                    new_creature.chromosome[random.randint(0, 6)] = random.randint(0, 99)

        # Add the new agent to the new population
        new_population.append(new_creature)

    # At the end you need to compute average fitness and return it along with your new population
    avg_fitness = np.mean(overall_fitness)
    fitness_list.append(avg_fitness)

    # To calculate and display the average fitness chart.
    game_count += 1
    if game_count == defaults.game_params['nGames']:
        plt.plot(fitness_list)
        plt.xlabel("Generation")
        plt.ylabel("Average Fitness")
        plt.title("Change in average Fitness")
        plt.savefig("fitness_chart.png")

    return new_population, avg_fitness
