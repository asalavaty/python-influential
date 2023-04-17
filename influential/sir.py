#! usr/bin/env python3

# Import the requirements
import igraph
from random import random, randint

# =============================================================================
#
#    Simulate SIR
#
# =============================================================================

def simulate_sir(graph, beta, gamma, no_sim = 100):

    """
    
    This function runs simulations for an SIR (susceptible-infected-recovered) model, on a graph. 
    The SIR model is a simple model from epidemiology that simulates the spread of disease in a comunity. 
    The individuals of the population might be in three states: susceptible, infected and recovered. 
    Recovered people are assumed to be immune to the disease. 
    Susceptibles become infected with a rate that depends on their number of infected neighbors. 
    Infected people become recovered with a constant rate. 
    The number of timesteps for performing each simulation is randomly selected from the 
    range of numbers between 1 and half of the number of network nodes (1/2*graph.vcount()).
    In this function, the initially infected individual is randomly selected from 
    all network vertices in each simulation round.

    :param graph: The graph to be evaluated.
    :type graph: A graph (network) of the igraph class (igraph.Graph).

    :param beta: The rate of infection of an individual that is susceptible and has a single infected neighbor (should be a positive scalar/float between 0 and 1). 
    The infection rate of a susceptible individual with n infected neighbors is n times beta. Formally this is the rate parameter of an exponential distribution.
    :type beta: float

    :param gamma: The rate of recovery of an infected individual (should be a positive scalar/float between 0 and 1). 
    Formally this is the rate parameter of an exponential distribution.
    :type gamma: float

    :param no_sim: The number simulation runs to perform.
    :type no_sim: int

    :return: A list of discotionaries with a length equal to the number of simulations (no_sim). Each dictionary 
    consists of three keys including Susceptible, Infected, and Recovered, and each key has values corresponding to 
    the timesteps of the respective simulation round.

    """

    # Get the number of vertices in the graph
    N = graph.vcount()

    final_result = []

    for i in range(no_sim):

        # Define times based on N
        times = randint(1, round(N/2))

        # Initialize the vertex attributes
        graph.vs["status"] = "S"  # All vertices are susceptible initially
        graph.vs[randint(0, N)]["status"] = "I"  # A random vertex is infected initially

        # Initialize the number of susceptible, infected, and recovered vertices
        S = [N - 1]
        I = [1]
        R = [0]

        # Run the simulation for times timesteps
        for t in range(times):
            # Get the number of infected vertices
            num_infected = sum(1 for v in graph.vs if v["status"] == "I")

            # Get the number of recovered vertices
            num_recovered = sum(1 for v in graph.vs if v["status"] == "R")

            # Update the number of susceptible, infected, and recovered vertices
            S.append(N - (num_infected + num_recovered))
            I.append(num_infected)
            R.append(num_recovered)

            # For each infected vertex, try to infect its neighbors
            for v in graph.vs.select(status="I"):
                neighbors = v.neighbors()
                for neighbor in neighbors:
                    if neighbor["status"] == "S" and random() < beta:
                        neighbor["status"] = "I"

            # For each infected vertex, try to recover
            for v in graph.vs.select(status="I"):
                if random() < gamma:
                    v["status"] = "R"
                    R[-1] += 1

            tmp_result = dict(Susceptible = S, Infected = I, Recovered = R)

        final_result.append(tmp_result)

    return final_result
