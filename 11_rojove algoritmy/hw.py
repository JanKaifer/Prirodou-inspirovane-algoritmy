# %% [markdown]
# # Domácí úkol
# Úkolem bylo zobecnit ukázkovou mravenčí kolonii, která řeší obchodníka, zobecnit pro libovolný počet aut (obchodníků).
# Následuje původní řešení, kde jsem vždy zmínil pomocí `# CHANGE:` jaké změny jsem udělal a proč.
#
# `# CHANGE:`
# Abychom mohli znovu-použít co nejvíce kódu, tak jsem si úlohu trochu transformoval (ekvivalentně).
# Mám jedno vozidlo a s ním se snažím dovézt všechny zásilky. Jsem ale omezen jeho kapacitou, tak si to musím dát na více okruhů.
# Zajímá nás totiž součet všech vzdáleností a počet aut (počet okruhů). Také se pro nás vrcholy budou až jednotlivé zásilky (které mohou sdílet souřadnice).
# To opět ničemu nebrání a dokonce se nám někdy může hodit dovézt do vrcholu pouze část zásilek (a ne všechy, třeba se všechny nevejdou do jenoho auta).
# Zároveň pro jednoduchost bude náš solution začínat i končit v depu.
# Nemohu ale použít feromony ve stejné podobě, protože potom mi problém zdegeneruje do podoby co balíček, to samostatný výlet.
# Proto budu mít feremony per-car.

# %%
from collections import namedtuple
import math
import functools
import numpy as np
import csv
import pprint
import time
import xml.etree.ElementTree as ET
from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib import collections as mc

# %% [markdown]
# ## Optimalizace mravenčí kolonií
#
# Optimalizace mravenčí kolonií (Ant Colony Optimization (ACO)) je algoritmus inspirovaný chováním mravenců při hledání potravy, který se hodí především pro kombinatorickou optimalizaci, konkrétně na problémy, které se dají převést na problém hledání cest v grafu.
#
# Mravenci se pohybují v prostředí a zanechávají za sebou feromonouvou stopu, která časem slábne a díky které spolu komunikují a slouží jim jako pozitivní druh zpětné vazby. Mravenec začne v náhodném bodě a rozhoduje se kam půjde dál. Nejprve se pohybují náhodně kolem mraveniště. Jakmile naleznou potravu, vrací se stejnou cestou, kterou k potravě došli a zanechávají za sebou feromonovou stopu. Jakmile nějkaý další mravenec narazí na feromonovou stopu, s větší pravděpodobností se po ní vydá spíše, než aby dále prozkoumával prostředí. Čím více mravenců se pohybuje mezi zdrojem potravy a mraveništěm, tím silnější je stopa a šance, že cesta přitáhne další mravence. Navíc na kratší cestě feromon vyprchává pomaleji, takže bude silnější a bude přitahovat více mravenců.
#
# Jen tak pro zajímavost rozhraní pro modelování různých přírodou inspirovaných systémů a algoritmů [NetLogo](http://ccl.northwestern.edu/netlogo/) poskytuje i simulaci mravenečků při hledání potravy. Podívat se na ně můžeme [zde](http://www.netlogoweb.org/launch#http://www.netlogoweb.org/assets/modelslib/Sample%20Models/Biology/Ants.nlogo).
#
# Zkusíme si s jeho pomocí vyřešit [Problém obchodního cestujícího](https://en.wikipedia.org/wiki/Travelling_salesman_problem), který se dá převést na problém hledání nejkratší Hamiltonovské kružnice v úplném grafu. Hamiltonovská kružnice v grafu je kružnice, která prochází všemi vrcholy. Implementace už je zde trochu složitější, ale pořád celkem rozumná.
#
# Nejprve si vytvoříme kolekci ```namedtuple```, do které si uložíme informace o vrcholu, tedy jeho souřadnice a název. Je vhodnější než normální třída, protože je to rychlejší struktura.
#
# `# CHANGE:` Změníme jméno ať to sedí naší interpretaci.
# Také si vyrobíme objekt, který bude reprezentovat nastavení našeho problému.

# %%
Package = namedtuple("Package", ["name", "vertex", "x", "y", "weight"])
Task = namedtuple("Task", ["packages", "depo", "max_load"])

# %% [markdown]
# Dále budeme potřebovat funkci, co nám spočítá vzdálenost dvou bodů. To uděláme chytře a použijeme k tomu lru (least recently used) cache, která si pamatuje si vyhodnocené funkce a při jejich opakovaném volání se stejnými parametry se nemusí znovu vyhodnocovat.

# %%
# @functools.lru_cache(maxsize=None) can't hash task
def distance(task, p1, p2):
    return math.sqrt((task.packages[p1].x - task.packages[p2].x) ** 2 + (task.packages[p1].y - task.packages[p2].y) ** 2)


# %% [markdown]
# Dále se bude hodit fitness funkce, která jen vyhodnotí danou cestu mezi městy tak, ze bere dvojice po sobě jdoucích vrcholů v řešení a sčítá vzdálenosti měst.
#
# `# CHANGE:`
# Naše solution bude moci navštívit depo vícekrát.
# Takto budeme moci potom jeden dlouhý sled rozdělit do několika kružnic, které budou značit trasy jednotlivých vozidel.

# %%
def fitness(task, distance, solution):
    cost = 0
    for trip in solution:
        assert trip[0] == task.depo
        assert trip[-1] == task.depo

        solution_distance = 0
        for x, y in zip(trip, trip[1:]):
            solution_distance += distance(task, x, y)
        cost += solution_distance
    return cost


# %% [markdown]
# Samotný algoritmus bude ještě potřebovat funkci na počáteční inicializaci feromonu, která by sice měla být stopa všude nulová, ale protože s ní pracujeme jako s pravděpodobností, tak by to nefungovalo, tak ji nastavíme na nějakou malou hodnotu. Také bude potřeba dělat update feromonu a to tak, že na všechny hrany v cestě rozpočítáme rovnoměrně tu fitness, která říká, jak byla váha dobrá. A protože délku chceme minimalizovat, takže použijeme inverzní Q/fit, kde Q bude nějaká konstanta.

# %%
def initialize_pheromone(N):
    return 0.01 * np.ones(shape=(N, N, N))


def update_pheromone(pheromones_array, solutions, fits, Q=100, rho=0.6):
    pheromone_update = np.zeros(shape=pheromones_array.shape)
    for solution, fit in zip(solutions, fits):
        for i, trip in enumerate(solution):
            for x, y in zip(trip, trip[1:]):
                pheromone_update[i][x][y] += Q / fit

    return (1 - rho) * pheromones_array + pheromone_update


# %% [markdown]
# Ještě nám zbývá pomocná funkce, kde mravenec generuje řešení, tedy náhodně prochází města, dokud neobejde všechny. Pak už můžeme napsat hlavní funkci algoritmu, kde se vytváří řešení, a podle jejich kvality se upravuje feromon na hranách. Zároveň si pamatujeme nejlepší řešení, abychom ho na konci mohli vrátit.
#
# `# CHANGE:`
# V generování možný řešení si musíme dát pozor na maximální nálož vozidel. Zásilky, které ji překročí budeme proto ignorovat.
# Zároveň musíme generování upravit pro podporu více tripů v řešení.

# %%
def generate_solutions(task, pheromones, distance, N, alpha=1, beta=3):
    # pravdepodobnost vyberu dalsiho mesta
    def compute_prob(trip_idx, v1, v2):
        # CHANGE: There will be many reqests with equal coordinates. We need to give them better than infinite probability.
        dist = distance(task, v1, v2)
        if dist < 1:
            dist = 2 - dist
        else:
            dist = 1 / dist
        tau = pheromones[trip_idx, v1, v2]
        ret = pow(tau, alpha) * pow(dist, beta)
        return ret if ret > 0.000001 else 0.000001

    pheromones_shape = pheromones.shape[0]
    for i in range(N):
        solution = []
        completed_packages = set([task.depo])

        while len(completed_packages) < len(task.packages):
            trip = [task.depo]
            space_left = task.max_load
            available = set(range(pheromones_shape))
            available.difference_update(completed_packages)

            while True:  # We will wait until we pick depo again
                for p in available:
                    if task.packages[p].weight > space_left:
                        available.remove(p)

                # We want to always consider depo as an option, but we need at least one package
                if trip[-1] != task.depo:
                    available.add(task.depo)

                probs = np.array(list(map(lambda x: compute_prob(len(solution), trip[-1], x), available)))
                selected = np.random.choice(list(available), p=probs / sum(probs))  # vyber hrany
                trip.append(selected)
                completed_packages.add(selected)
                available.remove(selected)

                if trip[-1] == task.depo:
                    break
            solution.append(trip)
        yield solution


# %% [markdown]
# Nyní už si můžeme vytvořit hlavní kód ACO.

# %%
def ant_solver(task, distance, ants=10, max_iterations=3000, alpha=1, beta=3, Q=100, rho=0.8):
    pheromones = initialize_pheromone(len(task.packages))
    best_solution = None
    best_fitness = float("inf")

    for i in range(max_iterations):
        solutions = list(generate_solutions(task, pheromones, distance, ants, alpha=alpha, beta=beta))
        fits = list(map(lambda x: fitness(task, distance, x), solutions))
        pheromones = update_pheromone(pheromones, solutions, fits, Q=Q, rho=rho)

        for s, f in zip(solutions, fits):
            if f < best_fitness:
                best_fitness = f
                best_solution = s

        print(f"{i:4}, {np.min(fits):.4f}, {np.mean(fits):.4f}, {np.max(fits):.4f}")
    return best_solution, pheromones


# %% [markdown]
# Zkusíme si nyní algoritmus otestovat na hlavních evropských městech, vstupní data jsou uložena v souboru *cities.csv*.

# `# CHANGE:` načítání vstupu už je úplně jiné

# %%
def load_task(file):
    tree = ET.parse(file)
    root = tree.getroot()

    assert root.tag == "instance"

    Node = namedtuple("Node", ["id", "x", "y"])

    def parse_node(node_tag):
        return Node(
            int(node_tag.attrib["id"]),
            float(node_tag.find("cx").text),
            float(node_tag.find("cy").text),
        )

    nodes_by_id = {}
    for node_tag in root.find("network").find("nodes"):
        node = parse_node(node_tag)
        nodes_by_id[node.id] = node

    def parse_package(package_tag):
        node = nodes_by_id[int(package_tag.attrib["node"])]
        return Package(
            package_tag.attrib["id"],
            node.id,
            node.x,
            node.y,
            float(package_tag.find("quantity").text),
        )

    packages = list(map(parse_package, root.find("requests")))

    vehicle_tag = root.find("fleet").find("vehicle_profile")
    depo_node = nodes_by_id[int(vehicle_tag.find("departure_node").text)]
    assert depo_node.id == int(vehicle_tag.find("arrival_node").text)
    max_load = float(vehicle_tag.find("capacity").text)

    packages.append(Package("depo", depo_node.id, depo_node.x, depo_node.y, 0))
    depo = len(packages) - 1

    return Task(packages, depo, max_load)


task = load_task("11_rojove algoritmy/domaci_ukol_data/data_32.xml")
best_solution, pheromones = ant_solver(task, distance, ants=10, max_iterations=1000)

# %% [markdown]
# Vykreslíme si nalezené řešení a množství feromononu na jednotlivých hranách. Feromon bude modrý, tloušťka čáry značí množství feromonu na hraně. Červenou barvou vykreslíme nejlepší řešení a vypíšeme si i jeho fitness a pořadí měst v něm. Odkomentováním zakomentované řádky si můžete vyzkoušet, jak různé nastavení alpha a beta ovlivňuje nalezená řešení.

# %%
def show_trip(task, solution, trip_idx):
    trip = solution[trip_idx]
    lines = []
    colors = []
    for i, v1 in enumerate(task.packages):
        for j, v2 in enumerate(task.packages):
            lines.append([(v1.x, v1.y), (v2.x, v2.y)])
            colors.append(pheromones[trip_idx][i][j])

    lc = mc.LineCollection(lines, linewidths=np.array(colors))

    plt.figure(figsize=(12, 8))
    ax = plt.gca()
    ax.add_collection(lc)
    ax.autoscale()

    print("Fitness: ", fitness(task, distance, solution))

    solution_vertices = [task.packages[i] for i in trip]
    pprint.pprint(solution_vertices)

    solution_lines = []
    for i, j in zip(trip, trip[1:]):
        solution_lines.append([(task.packages[i].x, task.packages[i].y), (task.packages[j].x, task.packages[j].y)])

    solutions_lc = mc.LineCollection(solution_lines, colors="red")
    ax.add_collection(solutions_lc)

    plt.show()


show_trip(task, best_solution, 0)

# %% [markdown]
# Pěkná simulace hledání nejkratší cesty v grafu se nachází [zde](http://thiagodnf.github.io/aco-simulator).
#
# ## Úkol na cvičení
#
# Poslední zmiňovaný algoritmus, který zde ale není naimplementovaný, je optimalizace pomocí včelí kolonie. Umělé včelí kolonie (ABC) je optimalizační algoritmus založený na chování včel při hledání potravy. Včely jsou rozděleny do třech skupin - na dělnice, vyčkávající včely a průzkumníky. Každá dělnice opracovává jeden zdroj jídla (a pozice těchto zdrojů kódují řešení). Při opracování dělnice navštíví zdroje jídla v okolí, a pokud je jiné řešení kvalitnější (má lepší fitness) nahradí svůj zdroj tímto novým zdrojem. Potom se všechny dělnice sejdou v úle, vymění si informace o kvalitě zdrojů a vyčkávající včely si vyberou některé z těchto zdrojů pomocí ruletové selekce. Dělnice si zároveň pamatují, jak dlouho už opracovávají daný zdroj, a pokud přesáhne tato doba nastavený limit, zdroj opustí a stanou se z nich průzkumníci. Průzkumníci prohledávají prostor náhodně a hledají nové zdroje potravy
#
# Zkuste si tedy naimplementovat ve zbytku cvičení optimalizaci pomocí včelí kolonie a vyřešit s ní třeba problém rastrigin funkce, který je výše vyřešený pomocí optimalizace hejna částic.
#
# ## Domácí úkol
#
# Za domácí úkol budete mít vyřešit pomocí optimalizace mravenčí kolonií [Vehicle Routing Problem](https://en.wikipedia.org/wiki/Vehicle_routing_problem), což je vlastně jen zobecněný problém obchodního cestujícího na princip optimalizace rozvozu zásilek doručovací společnosti. Jedná se o to, že máme depa, každé má svá vlastní vozidla s danou kapacitou a nějakou množinu zásilek, které je potřeba rozvézt k jejich majitelům. Cílem je najít množinu doručovacích tras tak, aby byly všechny zásilky dodány majitelům a aby byly minimalizované celkové náklady, tedy aby byl použit co nejmenší počet vozidel a aby byly trasy co nejkratší.
#
# V našem případě použijeme zjednodušenou verzi tohoto problému s jedním depem, které má neomezený počet vozidel jednoho typu. Vstupní data najdete ve složce *domaci_ukol_data*, jsou ve formátu xml a obsahují 3 soubory -- 2 malé a jeden větší, které zároveň obsahují:
# - Seznam uzlů se souřadnicemi x a y, kdy uzel s typem 0 je depo, a ty ostatní s typem 1 jsou lokace zákazníků.
# - Seznam vozidel, v tomto případě máme jeden typ vozidla, které musí začínat a končit v depu a má nějakou maximální kapacitu předmětů, které uveze.
# - Seznam požadavků, neboli do jakého uzlu se toho má co dovézt.
#
# Svůj kód, popis řešení, výsledky a jejich rozbor mi pošlete emailem do stanoveného deadline.

# %%
