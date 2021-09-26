import design_tools_template as dt
import numpy as np
from math import tan, fabs



def geometry(aircraft):

    Lv_bw = aircraft['geo_param']['EV']['Lb']
    ARw = aircraft['geo_param']['wing']['AR']
    Sw = aircraft['geo_param']['wing']['S']
    xr_w = aircraft['geo_param']['wing']['xr']
    zr_w = aircraft['geo_param']['wing']['zr']
    delta_w = aircraft['geo_param']['wing']['dihedral']
    lambda_w = aircraft['geo_param']['wing']['taper']
    sweep_w = aircraft['geo_param']['wing']['sweep']
    Cvt = aircraft['geo_param']['EV']['Cvt']
    ARv = aircraft['geo_param']['EV']['AR']
    lambda_v = aircraft['geo_param']['EV']['taper']
    zr_v = aircraft['geo_param']['EV']['zr']
    sweep_v = aircraft['geo_param']['EV']['sweep']


    dimensions = {}
    wing = {}
    EV = {}
    # parametros da asa

    bw = np.sqrt(ARw * Sw)
    wing['b'] = bw
    cr_w = 2 * Sw / (bw * (1 + lambda_w))
    wing['cr'] = cr_w
    ct_w = lambda_w * cr_w
    wing['ct'] = ct_w

    yt_w = bw / 2
    xt_w = xr_w + yt_w * tan(sweep_w) + (cr_w - ct_w)/4 
    wing['xt'] = xt_w
    wing['yt'] = yt_w
    zt_w = zr_w + yt_w * tan(delta_w)
    wing['zt'] = zt_w
    cm_w = 2 * cr_w * (1 + lambda_w + lambda_w**2)/(3 * (1 + lambda_w))
    wing['cm'] = cm_w
    ym_w = bw * (1 + 2 * lambda_w)/(6 * (1 + lambda_w))
    xm_w = xr_w + ym_w * tan(sweep_w) + (cr_w - cm_w)/4
    wing['xm'] = xm_w
    wing['ym'] = ym_w
    zm_w = zr_w + ym_w * tan(delta_w)
    wing['zm'] = zm_w

    # parametros da empenagem vertical

    Lv = Lv_bw * bw
    EV['L'] = Lv
    Sv = Sw * bw * Cvt / Lv
    EV['S'] = Sv

    bv = np.sqrt(ARv * Sv)
    EV['b'] = bv

    cr_v = 2 * Sv/(bv * (1 + lambda_v))
    EV['cr'] = cr_v

    ct_v = lambda_v * cr_v
    EV['ct'] = ct_v

    cm_v = 2 * cr_v * (1 + lambda_v + lambda_v**2)/(3 * (1 + lambda_v))
    xm_v = xm_w + Lv + (cm_w - cm_v)/4
    zm_v = zr_v + bv * (1 + 2 * lambda_v)/(3 * (1 + lambda_v))
    xr_v = xm_v - (zm_v - zr_v) * tan(sweep_v) + (cm_v - cr_v)/4
    EV['xr'] = xr_v

    zt_v = zr_v + bv
    xt_v = xr_v + (zt_v - zr_v) * tan(sweep_v) + (cr_v - ct_v)/4
    EV['xt'] = xt_v

    EV['zt'] = zt_v

    cm_v = (2 * cr_v/3) * (1 + lambda_v + lambda_v**2)/(1 + lambda_v)
    EV['cm'] = cm_v


    EV['xm'] =xm_v

    zm_v = zr_v + bv * (1 + 2 * lambda_v)/(3 * (1 + lambda_v))
    EV['zm'] = zm_v

    
    dimensions['wing'] = wing
    dimensions['EV'] = EV

    return dimensions

dimensions_test = {
            "wing": {
            "b": 28.074988869098416 ,
            "cr": 5.3933059334262 ,
            "ct": 1.267426894355157 ,
            "xt": 18.944010614572072 ,
            "yt": 14.037494434549208 ,
            "zt": 1.2281216273313065 ,
            "cm": 3.756317488774531 ,
            "xm": 15.659971822785682 ,
            "ym": 5.569532204800901 ,
            "zm": 0.4872709290626237
            },
            "EV": {
            "S": 14.959999999999999 ,
            "L": 15.44124387800413 ,
            "b": 4.358807176281144 ,
            "cr": 3.944978890651773 ,
            "ct": 2.919284379082312 ,
            "xr": 29.25388711043971 ,
            "xt": 33.299364009371466 ,
            "zt": 4.358807176281144 ,
            "cm": 3.4576757510555542 ,
            "xm": 31.17587613521955 ,
            "zm": 2.070850918999471
            }
        }


aircraft = dt.default_aircraft()

answer = geometry(aircraft)

print()
print("Parâmetros da asa")
print()

for key, value in dimensions_test["wing"].items():
    calculado = answer["wing"][key]
    diferenca = fabs(calculado - value)
    if diferenca < 1.0e-13:
        print(f"{key}: {value} -> ok")
    else:
        diferenca = fabs(calculado - value)
        print(f"{key} -> Valor esperado: {value}, valor recebido: {calculado}, diferença de {diferenca}")


print()
print("Parâmetros da empenagem vertical")
print()

answer = geometry(aircraft)
for key, value in dimensions_test["EV"].items():
    calculado = answer["EV"][key]
    diferenca = fabs(calculado - value)
    if diferenca < 1.0e-13:
        print(f"{key}: {value} -> ok")
    else:
        diferenca = fabs(calculado - value)
        print(f"{key} -> Valor esperado: {value}, valor recebido: {calculado}, diferença de {diferenca}")