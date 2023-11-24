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


cpd_c = TabularCPD(variable='C', variable_card=2, values=[[0.9995], [0.0005]]) # C=0 cutremurul nu are loc, C=1 invers

cpd_i = TabularCPD(variable='I', variable_card=2,
                   values=[[0.99, 0.97],
                           [0.01, 0.03]],
                    evidence=['C'],
                    evidence_card=[2]) # I=0 incendiul nu are loc, I=1 invers

cpd_a = TabularCPD(variable='A', variable_card=2,
                   values=[[0.9999, 0.98, 0.05, 0.02],
                           [0.0001, 0.02, 0.95, 0.98]],
                    evidence=['C', 'I'],
                    evidence_card=[2, 2]) # A=0 ialarma nu se declanseaza, A=1 invers 