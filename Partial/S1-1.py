import random

#Functie pentru aruncarea monezii in functie de probabilitatea data pentru a determina daca este stema sau nu
def aruncare_moneda(probabilitate_stema):
    if random.random() < probabilitate_stema:
        return 1
    else:
        return 0

#Funcatie pentru simularea unei instante a jocului
def simulate_game():
    #Se alege primul jucator: 0 - J0, 1 - J1
    primul_jucator = random.randint(0, 1)

    #Calculeaza numarul de steme pentru fiecare jucator
    if primul_jucator == 0:
        n = aruncare_moneda(1/2)
        m = 0
        for _ in range(n+1):
            m += aruncare_moneda(2/3)
    else:
        m = aruncare_moneda(2/3)
        n = 0
        for _ in range(n+1):
            n += aruncare_moneda(1/2)

    #Se verifica castigul
    if n >= m:
        return primul_jucator
    else:
        return 1 - primul_jucator

#Se simuleaza cele 10000 jocuri si se memoreza castigatorul pentru fiecare
rezultate = [simulate_game() for _ in range(10000)]

#Se calculeaza probabilitatea pentru fecare jucator, impartind numarul de aparitii la numarul de iteratii
print("Sanse castig J0:", rezultate.count(0) / 10000)
print("Sanse castig J1:", rezultate.count(1) / 10000)

#Rezultate

#1
#Sanse castig J0: 0.3916
#Sanse castig J1: 0.6084

#2
#Sanse castig J0: 0.391
#Sanse castig J1: 0.609

#3
#Sanse castig J0: 0.3882
#Sanse castig J1: 0.6118

#4
#Sanse castig J0: 0.392
#Sanse castig J1: 0.608

#5
#Sanse castig J0: 0.3897
#Sanse castig J1: 0.6103