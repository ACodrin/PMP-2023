from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import matplotlib.pyplot as plt
import networkx as nx

# J - primul jucator
# R1 - prima runda
# R2 - a doua runda
# C - castigator

model = BayesianNetwork([('J', 'R1'), ('J', 'R2'), ('R1', 'R2'), ('R1', 'C'), ('R2', 'C')])

cpd_J = TabularCPD(variable='J', variable_card=2, values=[[0.5], [0.5]])

cpd_R1 = TabularCPD(variable='R1', variable_card=2,
                    values=[[0.5, 1/3],
                            [0.5, 2/3]],
                    evidence=['J'],
                    evidence_card=[2])

cpd_R2 = TabularCPD(variable='R2', variable_card=3,
                    values=[[1/3,   0.5,    0.25, ],
                            [2/3,   0.5,    0.5, ],
                            [0,     0,      0.25, ]],
                    evidence=['J', 'R1'],
                    evidence_card=[2, 2])
