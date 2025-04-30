# -*- coding: utf-8 -*-
"""
Sistemas de Adquisición y Procesamiento de Señales
Facultad de Ingeniería - UNER

    Función que permite recuperar los parámetros de diseño de un 
    filtro generado con la herramienta pyFDA y guardado en 
    formato .npz.

Autor: Juan Ignacio Cerrudo
Fecha: Mayo 2020
"""
import numpy as np


def filter_parameters(fil):
    """
    ------------------------
    INPUT:
    --------
    fil: archivo .npz generado con pyFDA
    ------------------------
    OUTPUT:
    --------
    nada, devuelve por consola los parámetros principales    
    """
    fil = np.load(fil, allow_pickle=True)
    print("Frecuencia de muestreo: {:.1f} Hz".format(fil['f_S']))   
    print("Aproximación utilizada:" + fil['creator'][1])  
    tipo = fil['rt']
    if tipo == 'LP':
        print("Orden del filtro: {:}".format(fil['N']))
        print("Filtro Pasa-Bajos")
        print("Frecuencia de corte banda de paso : {:.2f} Hz".format(fil['F_PB']*fil['f_S']))
        print("Frecuencia de corte banda de rechazo: {:.2f} Hz".format(fil['F_SB']*fil['f_S']))
        
    if tipo == 'BP':
        if (fil['creator'][1] == 'pyfda.filter_designs.cheby1' or fil['creator'][1] == 'pyfda.filter_designs.cheby2' or fil['creator'][1] == 'pyfda.filter_designs.butter' or fil['creator'][1] == 'pyfda.filter_designs.bessel'):
            print("Orden del filtro: {:}".format(2*fil['N']))
        else:
            print("Orden del filtro: {:}".format(fil['N']))
        print("Filtro Pasa-Banda")
        print("Frecuencia de corte banda de rechazo inferior: {:.2f} Hz".format(fil['F_SB']*fil['f_S']))
        print("Frecuencia de corte inferior banda de paso : {:.2f} Hz".format(fil['F_PB']*fil['f_S']))
        print("Frecuencia de corte superior banda de paso : {:.2f} Hz".format(fil['F_PB2']*fil['f_S']))
        print("Frecuencia de corte banda de rechazo superior: {:.2f} Hz".format(fil['F_SB2']*fil['f_S']))

    if tipo == 'BS':
        print("Orden del filtro: {:}".format(2*fil['N']))
        print("Filtro Rechaza-Banda")
        print("Frecuencia de corte banda de paso inferior: {:.2f} Hz".format(fil['F_PB']*fil['f_S']))
        print("Frecuencia de corte inferior banda de rechazo : {:.2f} Hz".format(fil['F_SB']*fil['f_S']))
        print("Frecuencia de corte superior banda de rechazo : {:.2f} Hz".format(fil['F_SB2']*fil['f_S']))
        print("Frecuencia de corte banda de paso superior: {:.2f} Hz".format(fil['F_PB2']*fil['f_S']))
        
    if tipo == 'HP':
        print("Orden del filtro: {:}".format(fil['N']))
        print("Filtro Pasa-Alto")
        print("Frecuencia de corte banda de paso : {:.2f} Hz".format(fil['F_SB']*fil['f_S']))
        print("Frecuencia de corte banda de rechazo: {:.2f} Hz".format(fil['F_PB']*fil['f_S']))

#ejemplo
#filter_parameters('filtro_iir.npz')