"""
lighthouse_functions.py
=================================
Modul lighthouse_functions obsahuje funkce potrebne v main.py.

"""

import numpy as np
import math
import matplotlib.pyplot as plt

# skutecna poloha majaku v km
alfa_true = 5
beta_true = 0.5

# uvazovana oblast vyskytu majaku v rozsahu -10 az 10 km podel pobrezi
alfa = np.array(np.linspace(-10, 10, 201))
# a 10 m az 5 km do more (nula neni uvazovana kvuli pouziti funkce logaritmus)
beta = np.array(np.linspace(0.01, 5, 500))


# Uvazujeme priorni pravdepodobnost vyskytu majaku na souradnici alfa jako
# normalni rozdeleni se stredni hodnotou a_mean a smerodatnou odchylkou
# a_spread. Totez plati pro souradnici beta. Vse je v km. Tyto parametry
# urci uzivatel. Stejne tak urci pocet zablesku majaku, ze kterych se bude
# odhadovat skutecna poloha majaku.


def input_prior_parameters():
    """Nacte od uzivatele parametry priornich rozdeleni
    a pocet zablesku majaku. Vrati n-tici hodnot.

    """

    print('Vlozte parametry priorniho rozdeleni pravdepodobnosti '
          'polohy jednak podel pobrezi (alfa) a jednak smerem do more (beta). '
          'Obe rozdeleni jsou gaussovska, vlozte jejich stredni hodnotu '
          'a smerodatnou odchylku. Podmínky: -10 < alfa < 10 a 0 < beta < 5.')
    a_mean = float(input('Stredni hodnota souradnice alfa: '))
    if not -10 < a_mean < 10:
        raise Exception('Hodnota není v požadovaném rozsahu.')
    a_spread = float(input('Smerodatna odchylka souradnice alfa: '))
    if a_spread < 0.5:
        raise Exception('Hodnota musí být z výpočetních důvodů minimálně 0.5.')
    b_mean = float(input('Stredni hodnota souradnice beta: '))
    if not 0 < b_mean < 5:
        raise Exception('Hodnota není v požadovaném rozsahu.')
    b_spread = float(input('Smerodatna odchylka souradnice beta: '))
    if b_spread < 0.5:
        raise Exception('Hodnota musí být z výpočetních důvodů minimálně 0.5.')
    print('Vlozte pocet n zablesku majaku, ktery tvori Vas vzorek.'
          'Podmínka: n > 0.')
    num = float(input('Pocet vzorku: '))
    if not num.is_integer() or num <= 0:
        raise Exception('Zadejte přirozené číslo.')
    return a_mean, a_spread, b_mean, b_spread, int(num)


# z 1D normalnich rozdeleni se vytvori 2D priorni pravdepodobnosti rozdeleni


def compute_prior(a_mean, a_spread, b_mean, b_spread):
    """Z parametru 1D priornich rozdeleni vytvori 2D priorni rozdeleni
    typu np.array.

    """

    prior_alfa = (1 / math.sqrt(2 * math.pi) / a_spread *
                  np.exp(-0.5 * ((alfa - a_mean) / a_spread) ** 2))
    prior_beta = (1 / math.sqrt(2 * math.pi) / b_spread *
                  np.exp(-0.5 * ((beta - b_mean) / b_spread) ** 2))
    prior_unnorm = prior_alfa[np.newaxis, :] * prior_beta[:, np.newaxis]
    prior = prior_unnorm / np.sum(prior_unnorm)
    return prior


def post_likelihood():
    """Z predem zadanych hodnota a vstupu uzivatele vypocita
    posteriorni verohodnost typu np.array.

    """

    a_mean, a_spread, b_mean, b_spread, number_of_samples = input_prior_parameters()
    prior = compute_prior(a_mean, a_spread, b_mean, b_spread)
    # uniformni rozdeleni azimutu vyslanych zablesku...
    theta = np.pi * (np.random.rand(number_of_samples) - 0.5)
    # ...a odpovidajici pozice zasazeneho detektoru na pobrezi
    x = alfa_true + beta_true * np.tan(theta)

    # zaporne vzaty logaritmus verohodnostni funkce, kde verohodnostni funkce
    # je definovana jako soucin posteriornich pravdepodobnosti, ktere jsou soucinem
    # verohodnosti a priorni pravdepodobnosti. Pro odvozeni
    # viz http://www.di.fc.ul.pt/~jpn/r/bugs/lighthouse.html
    lkhd = -np.array([[sum(np.log(beta[j] / (math.pi * (beta[j] ** 2 +
                                                        (x - alfa[i]) ** 2)))) + np.log(prior[j, i])
                       for i in range(0, len(alfa))]
                      for j in range(0, len(beta))])
    return lkhd


# Graf:


def create_plot(lkhd):
    """Vykresli graf, kde na ose x je poloha majaku podel pobrezi
    a na ose y poloha majaku smerem do more. Zelene jsou vrstevnice
    verohodnostni funkce s vertikalnim rozestupem 2. Cervene poloha
    majaku, ktera slouzila k vygenerovani dat. Zeleny bod ukazuje
    odhadnutou polohu. Nejmensi vrstevnice odpovida intervalovemu
    odhadu polohy.

    """

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

    x, y = np.meshgrid(alfa, beta)
    levels = [np.amin(lkhd) + 2 * i for i in range(0, 7)]
    ax.contour(x, y, lkhd, levels, colors='g')

    ind = min_coord(lkhd)
    # bodovy odhad
    plt.scatter(alfa[ind[1]], beta[ind[0]], color='g')
    # skutecna hodnota
    plt.scatter(alfa_true, beta_true, color='r')

    ax.set_title('Zaporny logaritmus verohodnosti polohy majaku', fontsize=16)
    ax.set_xlabel(r'poloha podel pobrezi, $\alpha$ [km]', fontsize=12)
    ax.set_ylabel(r'vzdalenost od pobrezi, $\beta$ [km]', fontsize=12)
    ax.grid()
    plt.legend(['odhad', 'skutecnost'], fontsize=12)
    plt.show()


def min_coord(lkhd):
    """Najde souradnice minima funkce L.

    """

    ind = (np.unravel_index(np.argmin(lkhd, axis=None), lkhd.shape))
    return ind


def bounds_coord(lkhd):
    """Najde souradnice krajnich hodnot intervaloveho odhadu minima.

    """

    # najde rez o 2 vyse nez je minimum, tj. nejuzsi konturu v grafu
    ellipse = np.array((lkhd <= (np.amin(lkhd) + 2)))
    # najde nejvetsi a nejmensi alfa a beta v tomto rezu, coz jsou meze
    # intervaloveho odhadu
    maxmin = np.array(np.nonzero(ellipse))
    return maxmin


def report(lkhd):
    """Vypise intervalove a bodove odhady a vrati je jako seznam.

    """

    ind = min_coord(lkhd)
    maxmin = bounds_coord(lkhd)
    print('Odhad alfa je {:.1f} a lezi v intervalu {:.1f} az {:.1f}'.format(
        alfa[ind[1]], np.amin(alfa[maxmin[1, :]]), np.amax(alfa[maxmin[1, :]])))
    print('Odhad beta je {:.2f} a lezi v intervalu {:.2f} az {:.2f}'.format(
        beta[ind[0]], np.amin(beta[maxmin[0, :]]), np.amax(beta[maxmin[0, :]])))
    return [np.amin(alfa[maxmin[1, :]]), alfa[ind[1]], np.amax(alfa[maxmin[1, :]]),
            np.amin(beta[maxmin[0, :]]), beta[ind[0]], np.amax(beta[maxmin[0, :]])]
