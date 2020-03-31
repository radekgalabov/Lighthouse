"""
main.py
====================================
Program main.py řeší úlohu, kdy se hledá poloha majáku, který přestal fungovat
a náhodně vyzařuje úzce kolimované záblesky uniformně v čase a azimutu.
Maják se nachází na souřadnici alfa podél pobřeží a souřadnici beta
směrem do moře. Řešení se hladá metodou maximální věrohodnosti
(maximum likelihood estimate). Podél pobřeží jsou hustě umístěny detektory,
které neregistrují směr, nýbrž jen fakt, že k záblesku došlo. Nákres a potřebná
odvození případně zde: http://www.di.fc.ul.pt/~jpn/r/bugs/lighthouse.html

"""

import lighthouse_functions as lf
import numpy as np

# skutecna poloha majaku v km
alfa_true = 5
beta_true = 0.5

# uvazovana oblast vyskytu majaku v rozsahu -10 az 10 km podel pobrezi
alfa = np.array(np.linspace(-10, 10, 201))
# a 10 m az 5 km do more (nula neni uvazovana kvuli pouziti funkce logaritmus)
beta = np.array(np.linspace(0.01, 5, 500))

# zepta se na potrebne vstupy a spocte verohodnost
lkhd = lf.post_likelihood()

# vypise bodove a intervalove odhady skutecne polohy majaku
lf.report(lkhd)

print('Zavřete okno grafu a program se ukončí.')
# vykresli situaci do grafu
lf.create_plot(lkhd)
