import matplotlib.pyplot as plt
import scipy.stats as spy

lambda_clienti = 20 #clienti/ora

medie = 2  # minute
deviatie_standard = 0.5  # minute

alpha = 3  # minute (valoare aleatoare)

nr_ore_simulare = 24

nr_clienti_pe_ora = spy.poisson.rvs(lambda_clienti, size=nr_ore_simulare)
timp_plasare_plata = spy.norm.rvs(loc=medie, scale=deviatie_standard, size=nr_ore_simulare)
timp_gatit = spy.expon.rvs(scale=alpha, size=nr_ore_simulare)

timp_total = []
for i in range(nr_ore_simulare):
    timp_total.append(nr_clienti_pe_ora[i] * (timp_plasare_plata[i] + timp_gatit[i]))

plt.plot(range(nr_ore_simulare), timp_total)
plt.xlabel("Ora")
plt.ylabel("Timp total (min)")
plt.title("Timp total pentru fiecare ora")
plt.show()
