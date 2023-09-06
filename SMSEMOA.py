#
# Author: Diana Valencia
# Description: This code implements the SMS-EMOA 
#

import matplotlib.pyplot as plt 
import numpy as np
import time
from typing import Generator, List, TypeVar
from jmetal.algorithm.singleobjective.genetic_algorithm import GeneticAlgorithm
from jmetal.config import store
from jmetal.core.algorithm import Algorithm
from jmetal.core.operator import Crossover, Mutation, Selection
from jmetal.core.problem import Problem
from jmetal.operator import BinaryTournamentSelection
from jmetal.util.comparator import Comparator
from jmetal.util.evaluator import Evaluator
from jmetal.util.ranking import FastNonDominatedRanking
from jmetal.util.termination_criterion import TerminationCriterion
from jmetal.core.quality_indicator import HyperVolume
from sklearn.preprocessing import MinMaxScaler

S = TypeVar("S")
R = TypeVar("R")


class SMSEMOA(GeneticAlgorithm[S, R]):
    def __init__(
        self,
        problem: Problem,
        population_size: int,
        offspring_population_size: int,
        mutation: Mutation,
        crossover: Crossover,
        selection: Selection = BinaryTournamentSelection(FastNonDominatedRanking.get_comparator()),
        termination_criterion: TerminationCriterion = store.default_termination_criteria,
        population_generator: Generator = store.default_generator,
        population_evaluator: Evaluator = store.default_evaluator,
        dominance_comparator: Comparator = store.default_comparator,
    ):
        super(SMSEMOA, self).__init__(
            problem=problem,
            population_size=population_size,
            offspring_population_size=offspring_population_size,
            mutation=mutation,
            crossover=crossover,
            selection=selection,
            termination_criterion=termination_criterion,
            population_evaluator=population_evaluator,
            population_generator=population_generator,
        )
        self.dominance_comparator = dominance_comparator
        self.gen = 0
        self.ranking = FastNonDominatedRanking(self.dominance_comparator) 
        self.scaler = MinMaxScaler()

    #
    #  Description: execute the SMS-EMOA 
    #

    def run(self):
        self.gen = 0
        self.solutions = self.create_initial_solutions()
        self.solutions = self.evaluate(self.solutions)

        self.init_progress()
        while not self.stopping_condition_is_met():
            mating_population = self.selection(self.solutions)
            offspring_population = self.reproduction(mating_population)
            offspring_population = self.evaluate(offspring_population)
            self.solutions = self.replacement(self.solutions, offspring_population)
            self.update_progress()
            self.gen += 1
            if self.gen % 50 == 0:
                self.pop = self.plot_pop(self.solutions) 
    #
    # Description: Save the objective values of the individuals in a numpy array
    # Parameters: A population (pop)
    #
    def obtain_objs_list(self, pop ):
    	data = [ ]
    	for ind in pop:
    		data.append( ind.objectives )
    	return np.array(data)

    #
    # Description: Compute the hipervolume contribution of each individual
    # Parameters: The initialized hypervolume object (hv), and the normalized objective values (data_norm)
    #
    def compute_contributions(self, hv, data_norm):
    	data_size = len(data_norm)
    	mask = np.ones(data_size, dtype=bool)
    	hv_all = hv.compute( data_norm )
    	contributions = np.zeros( data_size )
    	for idx in range(data_size):
    		mask[idx] = False
    		contributions[idx] = hv_all - hv.compute(data_norm[mask])
    		mask[idx] = True
    	return contributions

    #
    # Description: Perform the survival selection
    # Parameters: Parent population (population) and Offspring population (offspring_population)
    #
    def replacement(self, population: List[S], offspring_population: List[S]) -> List[List[S]]:
        join_pop = population + offspring_population
        
        # Limits of the current population
        data_jpop = self.obtain_objs_list(join_pop)
        self.scaler.fit(data_jpop)
        
        # Fast non-dominated sorting
        front = self.ranking.compute_ranking(join_pop)

        # Obtain last front
        idx_lasfront = self.ranking.get_number_of_subfronts()-1
        lastfront = self.ranking.get_subfront(idx_lasfront)

        if(len(lastfront) == 1):
        	solutions =  [item for sublist in front[:-1] for item in sublist]
        else:
        	data = self.obtain_objs_list( lastfront )
        	data_norm =  self.scaler.transform( data )

        	# The reference point is [1.1]^m since the data is normalized
        	ref_pt = np.ones( self.problem.number_of_objectives ) * 1.1

        	#initialized the hypervolume using the given reference point
        	hv = HyperVolume(ref_pt)

        	# Obtain the lowest contribution the last front
        	contributions = self.compute_contributions(hv, data_norm)
        	idx_min = np.argmin( contributions )

        	solutions = [item for sublist in front[:-1] for item in sublist]
        	for i in range(len(data)):
        		if i != idx_min:
        			solutions.append( lastfront[i] )
        return solutions

    #
    # Description: Plot the objective values of the population 
    # Parameters: A population (pop)
    #
    def plot_pop(self, pop ):
        data = self.obtain_objs_list( pop )
        plt.plot(data[:,0],data[:,1], 'ro')
        plt.draw()
        plt.pause(0.01)
        plt.clf()


    #
    # Description: Obtain the final solutions
    #
    def get_result(self) -> R:
        return self.solutions

    #
    # Description: Obtain the name of the algorithm
    #
    def get_name(self) -> str:
        return "SMSEMOA"