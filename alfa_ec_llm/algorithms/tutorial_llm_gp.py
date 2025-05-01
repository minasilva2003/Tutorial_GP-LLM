import collections
import logging
import random
import time
from typing import Any, Dict, List, Optional, Tuple

from alfa_ec_llm.algorithms.evolutionary_algorithm import (
    EvolutionaryAlgorithm, FitnessFunction, Individual, Population)
from alfa_ec_llm.utils.openai_interface import OpenAIInterface, DeepSeekInterface
from alfa_ec_llm.utils.utils import (get_fitness_function,
                                     write_run_output_gp_plus_llm)

""" Implementation of Genetic Programming(GP) plus Large Language
Model, the purpose of this code is to describe how the algorithm
works. The intended use is for teaching. The design is supposed to be
simple.
"""

DEBUG = True

class TutorialLLMGPMuXo(EvolutionaryAlgorithm):

    def mutation(
        self,
        individual: Individual,
        fitness_function: FitnessFunction,
        llm_interface: DeepSeekInterface,
        generation_history: List[Tuple[str, str]],
        mutation_probability: float,
        samples: Optional[List[Any]] = None,
    ) -> List[Individual]:
        """
        Return a mutated individual
        """
        new_individual = Individual(individual.genome)
        new_individual.phenotype = individual.phenotype
        if random.random() < mutation_probability:
            prompt = fitness_function.form_prompt_rephrase_mutation(
                individual.phenotype, samples
            )
            response = llm_interface.predict_text_logged(prompt, temp=1)
            response["operation"] = "mutation"
            generation_history.append(response)
            phenotype = fitness_function.check_response_rephrase_mutation(
                response["content"], individual.phenotype
            )
            new_individual.phenotype = phenotype

        return new_individual
    
    def batch_mutation(
        self,
        individual_list: List[Individual],
        fitness_function: FitnessFunction,
        llm_interface: DeepSeekInterface,
        generation_history: List[Tuple[str, str]],
        mutation_probability: float,
        samples: Optional[List[Any]] = None,
    ) -> List[Individual]:

        clone_list = []
        mutant_list = []
        mutant_phenotype_list = []

        # Create clones of individuals -- some will be mutated and some won'
        for individual in individual_list:

            # Create clone
            new_individual = Individual(individual.genome)
            new_individual.phenotype = individual.phenotype
            
            # Decide whether clone will be mutated or won't
            if random.random() < mutation_probability:
                mutant_list.append(new_individual)
                mutant_phenotype_list.append(new_individual.phenotype)
            else:
                clone_list.append(new_individual)
        
        # Mutate all mutants in batch
        if len(mutant_list) > 0:

            # Generate prompt for batch mutation
            prompt = fitness_function.form_prompt_rephrase_batch_mutation(mutant_phenotype_list,
                                                                          samples)
        
            response = llm_interface.predict_text_logged(prompt, temp=1)
            response["operation"] = "mutation"
            generation_history.append(response)

            new_phenotypes = fitness_function.check_response_rephrase_batch_mutation(
                response["content"], individual.phenotype
            )
            
            # Attribute mutated phenotypes to mutants
            for new_phenotype, individual in zip(new_phenotypes, mutant_list):
                individual.phenotype = new_phenotype
                
        # Return clones and mutants
        return mutant_list+clone_list
    

    def batch_crossover(
        self,
        parent_pairs,
        fitness_function: FitnessFunction,
        llm_interface: DeepSeekInterface,
        generation_history: List[Tuple[str, str]],
        crossover_probability: float,
        samples: Optional[List[Any]] = None,
    ) -> List[Individual]:
        
        clone_children = []
        crossover_children = []
        crossover_children_phenotypes = []

        # Create children for each parent pair -- they start off as clones of the parents
        for parent_pair in parent_pairs:
            child0 = Individual(parent_pair[0].genome)
            child1 = Individual(parent_pair[1].genome)
            child0.phenotype = parent_pair[0].phenotype
            child1.phenotype = parent_pair[1].phenotype

            # Decide whether children pair will be crossed over or not
            if random.random() < crossover_probability:
                crossover_children.append([child0, child1])
                crossover_children_phenotypes.append([child0.phenotype, child1.phenotype])
            else:
                clone_children.append(child0)
                clone_children.append(child1)

        ## Do  batch crossover through LLM
        if len(crossover_children) > 0:
             
            # Generate prompt for batch crossover
            prompt = fitness_function.form_prompt_batch_crossover(
                crossover_children_phenotypes, samples
            )
     
            response = llm_interface.predict_text_logged(prompt, temp=1)
            response["operation"] = "crossover"
            generation_history.append(response)
           
            # Check whether phenotypes were correctly crossed over
            # If not, use the original phenotypes
            try:
                new_phenotype_pairs = fitness_function.check_response_batch_crossover(
                    response["content"], crossover_children)
                
            except AssertionError as e:
                new_phenotype_pairs = crossover_children_phenotypes
                logging.error(
                    f"{e} from formatting response for crossover for {response['content']} given {parent_pairs}"
                )

            # Assign new phenotypes to children
            for child_pair, phenotype_pair in zip(crossover_children, new_phenotype_pairs):
                child_pair[0].phenotype = phenotype_pair[0]
                child_pair[1].phenotype = phenotype_pair[1]
                clone_children.append(child_pair[0])
                clone_children.append(child_pair[1])
            
        return clone_children
 


    def crossover(
        self,
        parents: List[Individual],
        fitness_function: FitnessFunction,
        llm_interface: DeepSeekInterface,
        generation_history: List[Tuple[str, str]],
        crossover_probability: float,
        samples: Optional[List[Any]] = None,
    ) -> List[Individual]:
        """
        Return a crossed over individuals
        """
        children = []
        for individual in parents:
            new_individual = Individual(individual.genome)
            new_individual.phenotype = individual.phenotype
            children.append(new_individual)

        if random.random() < crossover_probability:
            prompt = fitness_function.form_prompt_crossover(
                [_.phenotype for _ in parents], samples
            )
            response = llm_interface.predict_text_logged(prompt, temp=1)
            response["operation"] = "crossover"
            generation_history.append(response)
            try:
                phenotypes = fitness_function.check_response_crossover(
                    response["content"], parents
                )
            except AssertionError as e:
                phenotypes = [_.phenotype for _ in parents]
                logging.error(
                    f"{e} from formatting response for crossover for {response['content']} given {phenotypes}"
                )

            for child, phenotype in zip(children, phenotypes):
                child.phenotype = phenotype

        return children
    

    def initialize_population_in_batch(
        self,
        fitness_function: FitnessFunction,
        param: Dict[str, Any],
        llm_interface: DeepSeekInterface,
        generation_history: List[Tuple[str, str]],
    ) -> List[Individual]:
        """
        LLM generates random individuals in batch based on zero-shot (no additional information to the prompt) prompt.
        """

        # Create prompt for batch initialization
        prompt = fitness_function.form_prompt_batch_individual_generation(param["population_size"])

        new_phenotypes = None

        # Loop until we get a valid batch of phenotypes
        while new_phenotypes is None or len(new_phenotypes) != param["population_size"]:

            # Get batch of randomly generated phenotypes
            response = llm_interface.predict_text_logged(prompt, temp=1)
            response["operation"] = "initialize_population"
            generation_history.append(response)
            
            new_phenotypes = (
                fitness_function.check_response_batch_individual_generation(param["population_size"],
                    response["content"]
                )
            )

        # Create individuals from the new phenotypes
        new_individuals = []
        for phenotype in new_phenotypes:
            new_individual = Individual(None)
            new_individual.phenotype = phenotype
            new_individuals.append(new_individual)
    
        return new_individuals
    

    def initialize_population(
        self,
        fitness_function: FitnessFunction,
        param: Dict[str, Any],
        llm_interface: DeepSeekInterface,
        generation_history: List[Tuple[str, str]],
    ) -> List[Individual]:
        """
        LLM generates random individuals based on zero-shot (no additional information to the prompt) prompt.
        """

        individuals = []
        for i in range(param["population_size"]):
            individual = Individual(None)
            prompt = fitness_function.form_prompt_individual_generation()
            response = llm_interface.predict_text_logged(prompt, temp=1)
            response["operation"] = "initialize_population"
            generation_history.append(response)
            individual.phenotype = (
                fitness_function.check_response_individual_generation(
                    response["content"]
                )
            )
            # Append the individual to the population
            individuals.append(individual)

        return individuals

    def run(self, param: Dict[str, Any]) -> Individual:
        """
        Return the best solution. Create an initial
        population. Perform an evolutionary search.
        """
        if "seed" not in param.keys():
            param["seed"] = int(time.time())

        random.seed(param["seed"])
        logging.info(f"Setting random seed: {param['seed']} {random.random():.5f}")
        fitness_function = get_fitness_function(param["fitness_function"])

        #local deepseek model
        llm_interface = DeepSeekInterface()
        param["llm_interface"] = llm_interface
        generation_history = []
        param["generation_history"] = generation_history

        # Create population in batch or individual by individual
        if param["batch"]:
            individuals = self.initialize_population_in_batch(
                fitness_function, param, llm_interface, generation_history
            )
        else:
            individuals = self.initialize_population(
                fitness_function, param, llm_interface, generation_history
            )

        if DEBUG:
            print("Number of individuals: ", len(individuals))

        population = Population(fitness_function, individuals)
        # Start evolutionary search
        best_ever = self.search_loop(population, param)

        return best_ever

    def search_loop(
        self, population: Population, param: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Return the best individual from the evolutionary search
        loop. Starting from the initial population.
        """

        param["cache"] = collections.OrderedDict()
        start_time = time.time()
        stats: Dict[str, List[Any]] = collections.defaultdict(list)
        llm_interface = param["llm_interface"]
        generation_history = param["generation_history"]
        fitness_function = population.fitness_function

        ######################
        # Evaluate fitness
        ######################
        self.evaluate_fitness(
            population.individuals,
            fitness_function,
            param,
        )
        # Print the stats of the population
        self.print_stats(0, population.individuals, stats, start_time)
        # Set best solution
        population.individuals = self.sort_population(population.individuals)
        best_ever = population.individuals[0]

        ######################
        # Generation loop
        ######################
        generation = 1
        while generation < param["generations"]:

            if DEBUG:
                print(f"Generation {generation} has {len(population.individuals)} individuals")

            new_individuals = []
            ##################
            # Selection
            ##################
             
            parents = self.tournament_selection(
                population.individuals,
                param["population_size"],
                param["tournament_size"],
            )

            if DEBUG:
                print("Parents:")
                for parent in parents:
                    print(parent.phenotype)
            ##################
            # Variation. Generate new individual solutions
            ##################

            # Genetic operators done by batch
            if param["batch"]:

                ##################
                # BATCH CROSSOVER
                ##################

                # Select number of parent pairs to choose for crossover. Handles uneven population
                if param["population_size"] % 2 == 0:
                    n_pairs = param["population_size"] // 2
                else:
                    n_pairs = (param["population_size"] + 1) // 2

                parent_pairs = []

                # Select parent pairs
                for _ in range (n_pairs):
                    pair = random.sample(parents,2)
                    parent_pairs.append(pair)
                    print(pair[0].phenotype, pair[1].phenotype)

                # Do crossover through LLM
                children = self.batch_crossover(
                        parent_pairs,
                        fitness_function,
                        llm_interface,
                        generation_history,
                        param["crossover_probability"],
                        param["cache"],
                    )
                
                # Add children to new individuals
                for child in children:
                    new_individuals.append(child)

                # truncate to population size
                new_individuals = new_individuals[: param["population_size"]]

                ##################
                # BATCH MUTATION
                ##################
                new_individuals = self.batch_mutation(
                                    new_individuals,
                                    fitness_function,
                                    llm_interface,
                                    generation_history,
                                    param["mutation_probability"],
                                    param["cache"],
                                )

            # genetic operators done by individual (batch = false)
            else:

                ##################
                # INDIVIDUAL CROSSOVER
                ##################
                while len(new_individuals) < param["population_size"]:
                    # Select parents
                    _parents = random.sample(parents, 2)
                    # Generate children by crossing over the parents
                    children = self.crossover(
                        _parents,
                        fitness_function,
                        llm_interface,
                        generation_history,
                        param["crossover_probability"],
                        param["cache"],
                    )
                    # Append the children to the new population
                    for child in children:
                        new_individuals.append(child)
           
                ## truncate to population size
                new_individuals = new_individuals[: param["population_size"]]

                ##################
                # INDIVIDUAL MUTATION
                ##################
            
                for i in range(len(new_individuals)):
                    new_individuals[i] = self.mutation(
                        new_individuals[i],
                        fitness_function,
                        llm_interface,
                        generation_history,
                        param["mutation_probability"],
                        param["cache"],
                    )

            ##################
            # Evaluate fitness
            ##################
            self.evaluate_fitness(new_individuals, fitness_function, param)

            ##################
            # Replacement. Replace individual solutions in the population
            ##################
            population.individuals = self.generational_replacement(
                new_individuals,
                population.individuals,
                elite_size=param["elite_size"],
                population_size=param["population_size"],
            )

            # Set best solution
            population.individuals = self.sort_population(
                population.individuals,
            )
            best_ever = population.individuals[0]

            # Print the stats of the population
            self.print_stats(generation, population.individuals, stats, start_time)

            # Increase the generation counter
            generation += 1

        write_run_output_gp_plus_llm(generation, stats, param, generation_history)
        return best_ever
