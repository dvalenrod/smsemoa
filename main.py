#
# Author: Diana Valencia
# Description: This code evaluate the SMSEMOA in the ZDT1 problem
#

from jmetal.operator import SBXCrossover, PolynomialMutation
from jmetal.problem import ZDT1
from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.util.solution import get_non_dominated_solutions, print_function_values_to_file,print_variables_to_file
from SMSEMOA import *
import sys


def main():
    problem = ZDT1()

    algorithm = SMSEMOA(
        problem=problem,
        population_size=25,
        offspring_population_size=1,
        mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20),
        crossover=SBXCrossover(probability=1.0, distribution_index=20),
        termination_criterion=StoppingByEvaluations(max_evaluations=10000)
    )

    algorithm.run()

    # Save the result of the algorithm 
    front = get_non_dominated_solutions(algorithm.get_result())
    print_function_values_to_file(front, 'FUN.SMSEMOA.ZDT1')
    print_variables_to_file(front, 'VAR.SMSEMOA.ZDT1')
main()