from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import matplotlib.pyplot as plt
import networkx as nx

# Definirea rețelei
model = BayesianNetwork([('Jucator', 'Stema0'), ('Jucator', 'Stema1'), ('Stema0', 'Castig')])

# Definirea Tabelului CPD pentru Jucator
cpd_jucator = TabularCPD(variable='Jucator', variable_card=2, values=[[0.5, 0.5]])

# Definirea Tabelului CPD pentru Stema0
cpd_stema0 = TabularCPD(variable='Stema0', variable_card=2, values=[[0.5, 0.5]])

# Definirea Tabelului CPD pentru Stema1 în funcție de Jucator și Stema0
cpd_stema1 = TabularCPD(variable='Stema1', variable_card=2, evidence=['Jucator', 'Stema0'],
                        evidence_card=[2, 2], values=[[2/3, 1/3, 0.5, 0.5]])

# Definirea Tabelului CPD pentru Castig în funcție de Stema0 și Stema1
cpd_castig = TabularCPD(variable='Castig', variable_card=2, evidence=['Stema0', 'Stema1'],
                        evidence_card=[2, 2], values=[[1, 0, 0, 1], [0, 1, 1, 0]])

# Adăugarea CPD-urilor la model
model.add_cpds(cpd_jucator, cpd_stema0, cpd_stema1, cpd_castig)

# Desenarea rețelei
nx.draw(model, with_labels=True, font_weight='bold')
plt.show()
