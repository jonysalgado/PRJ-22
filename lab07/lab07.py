# Setup

import design_tools_template as dt
import numpy as np
import matplotlib.pyplot as plt


# ===========================================================================
# Exercício


# Inicializando vetores
sweeps = np.arange(start=0, stop=45, step=0.1)*np.pi/180
vec_sm = np.zeros((2, sweeps.shape[0]))

# parâmetros padrões
nacelle = {
    'yn': 2.6,
    'zn': 0.0,
    'Ln': 4.3,
    'Dn': 1.5,
    'xn': 23.2
}
fus =  {
    'xcg': 16.4,
    'xnp': 16.9,
    'Lf': 32.8,
    'Df': 3.3
}
gravity = 9.81
W0_guess = 43090 * gravity
T0_guess : 125600
Mach_cruise = 0.77
altitude_cruise = 11000
range_cruise = 2390000.00000000000000
Mach_altcruise = 0.4
range_altcruise = 370000
altitude_altcruise = 4572
loiter_time = 2700
altitude_takeoff = 0
distance_takeoff = 1520
TO_flap_def = 0.34906585039887
TO_slat_def = 0
altitude_landing = 0
distance_landing = 1520
LD_flap_def = 0.69813170079773
LD_slat_def = 0
MLW_frac = 0.84


for i in range(sweeps.shape[0]):
    aircraft = dt.default_aircraft()
    dimensions = dt.geometry(aircraft)
    dimensions['nacelle'] = nacelle
    dimensions['fus'] = fus
    dimensions['ldg'] = aircraft['dimensions']['ldg']
    aircraft['dimensions'] = dimensions
    aircraft['geo_param']['wing']['sweep'] = sweeps[i]
    params = dt.analyze(aircraft, W0_guess, T0_guess,
                Mach_cruise, altitude_cruise, range_cruise,
                Mach_altcruise, range_altcruise, altitude_altcruise,
                loiter_time, altitude_takeoff, distance_takeoff, TO_flap_def, TO_slat_def,
                altitude_landing, distance_landing, LD_flap_def, LD_slat_def,
                MLW_frac)
    
    vec_sm[0, i] = params[5]
    vec_sm[1, i] = params[4]

# Primeira figura, sem a linha vertical
plt.figure(figsize=(12,8))
plt.grid(True)
plt.plot(vec_sm[0, :], sweeps*180/np.pi, 'b', label='Magem estática traseira')
plt.plot(vec_sm[1, :], sweeps*180/np.pi, 'k', label='Magem estática dianteira')
plt.vlines(0.05, ymin=-1, ymax=50.0, colors='r', linestyles='dashed', label='Limite traseiro')
plt.vlines(0.30, ymin=-1, ymax=50.0, colors='r', linestyles='dotted', label='Limite dianteiro')
plt.ylim(0, 45)
plt.xlim(-0.4, 0.9)
plt.xlabel("Margem estática")
plt.ylabel("Ângulo de enflechamento $\Lambda$ (°)")
plt.title("Limites de margem estática dianteiro e traseiro em função do ângulo de enflechamento de asa")
plt.legend()
plt.savefig("fig1.png", dpi=200)
plt.show()

# Segunda figura, com a linha vertical
plt.figure(figsize=(12,8))
plt.grid(True)
plt.plot(vec_sm[0, :], sweeps*180/np.pi, 'b', label='Magem estática traseira')
plt.plot(vec_sm[1, :], sweeps*180/np.pi, 'k', label='Magem estática dianteira')
plt.vlines(0.05, ymin=-1, ymax=50.0, colors='r', linestyles='dashed', label='Limite traseiro')
plt.vlines(0.30, ymin=-1, ymax=50.0, colors='r', linestyles='dotted', label='Limite dianteiro')
plt.hlines(17.45, xmin=-0.5, xmax=1.1, colors='g', linestyles='dotted', label='Caso de teste')
plt.ylim(0, 45)
plt.xlim(-0.4, 0.9)
plt.xlabel("Margem estática")
plt.ylabel("Ângulo de enflechamento $\Lambda$ (°)")
plt.title("Limites de margem estática dianteiro e traseiro em função do ângulo de enflechamento de asa")
plt.legend()
plt.savefig("fig2.png", dpi=200)
plt.show()


# Encontrando a fração de cg do allelse adequada
aircraft['weights']['per_xcg_allelse']
vec_allelse = np.arange(0.4, 0.5, 0.01)
for i in range(vec_allelse.shape[0]):
    aircraft = dt.default_aircraft()
    dimensions['nacelle'] = nacelle
    dimensions['fus'] = fus
    dimensions['ldg'] = aircraft['dimensions']['ldg']
    aircraft['dimensions'] = dimensions
    aircraft['weights']['per_xcg_allelse'] = vec_allelse[i]
    params = dt.analyze(aircraft, W0_guess, T0_guess,
                Mach_cruise, altitude_cruise, range_cruise,
                Mach_altcruise, range_altcruise, altitude_altcruise,
                loiter_time, altitude_takeoff, distance_takeoff, TO_flap_def, TO_slat_def,
                altitude_landing, distance_landing, LD_flap_def, LD_slat_def,
                MLW_frac)

    if params[5] >= 0.05 and params[4] <= 0.3:
        print(f'xcg_allelse: {vec_allelse[i]}')