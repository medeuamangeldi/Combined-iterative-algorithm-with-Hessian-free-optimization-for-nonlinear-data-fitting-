import pandas as pd
import numpy as np
import random
from pylab import *
from scipy.optimize import curve_fit
import plotly.graph_objects as go
import matplotlib.pyplot as plt
!pip install lmfit
!pip install numdifftools
import numdifftools
from lmfit import Model
import sklearn.metrics as sk
from scipy.interpolate import interp1d
from scipy.interpolate import CubicSpline, Akima1DInterpolator, PchipInterpolator, interp1d, InterpolatedUnivariateSpline

from matplotlib import cm
from matplotlib.ticker import LinearLocator

import warnings
warnings.filterwarnings("ignore")

"""#Upload and organize the data"""

plt.rcParams["figure.figsize"] = (8,6)
matplotlib.rcParams["figure.dpi"] = 300
plt.style.use('seaborn-ticks')
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
matplotlib.rcParams['lines.markersize'] = 7

def remove_outliers(sr, v):
  IQR = np.percentile(v, 75) - np.percentile(v, 25)

  lbound = np.percentile(v, 25) - 1.5*IQR
  ubound = np.percentile(v, 75) + 1.5*IQR

  list = []
  for i in range(len(v)):
    if v[i] < lbound or v[i]>ubound:
      list.append(i)
  srnew = np.delete(sr, list)
  vnew = np.delete(v, list)
  return srnew, vnew

def get_data(x,y):
  ss = np.array(x)
  vv = np.array(y/1000)

  ss = ss[np.logical_not(np.isnan(ss))]
  vv = vv[np.logical_not(np.isnan(vv))]

  s, v = remove_outliers(ss, vv)

  return np.log10(s), np.log10(v)

aa = pd.read_excel('/content/drive/MyDrive/Projects/Stat analysis extrusion research/new idea comparative study/Songy nuska.xlsx')
bb = pd.read_excel('/content/drive/MyDrive/Projects/Stat analysis extrusion research/new idea comparative study/Songy nuska.xlsx', sheet_name=1)
cc = pd.read_excel('/content/drive/MyDrive/Projects/Stat analysis extrusion research/new idea comparative study/Songy nuska.xlsx', sheet_name=2)
dd = pd.read_excel('/content/drive/MyDrive/Projects/Stat analysis extrusion research/new idea comparative study/Songy nuska.xlsx', sheet_name=3)
ee = pd.read_excel('/content/drive/MyDrive/Projects/Stat analysis extrusion research/new idea comparative study/Songy nuska.xlsx', sheet_name=4)
ff = pd.read_excel('/content/drive/MyDrive/Projects/Stat analysis extrusion research/new idea comparative study/Songy nuska.xlsx', sheet_name=5)

bb = bb.drop([0])
cc = cc.drop([0])
dd = dd.drop([2])
ee = ee.drop([0])
ff = ff.drop([0, 1, 4])

shear1, visc1 = get_data(bb['Angular Frequency'], bb['Complex Viscosity'])
shear2, visc2 = get_data(bb['Angular Frequency.1'], bb['Complex Viscosity.1'])
shear3, visc3 = get_data(bb['Angular Frequency.2'], bb['Complex Viscosity.2'])
shear4, visc4 = get_data(bb['Angular Frequency.3'], bb['Complex Viscosity.3'])
shear5, visc5 = get_data(bb['Angular Frequency.4'], bb['Complex Viscosity.4'])
shear6, visc6 = get_data(bb['Angular Frequency.5'], bb['Complex Viscosity.5'])

shear = [shear1, shear2, shear3, shear4, shear5, shear6]
visc = [visc1, visc2, visc3, visc4, visc5, visc6]

#data = [aa, bb, cc, dd, ee, ff]
temperature = [433.15, 453.15, 473.15, 493.15, 513.15, 533.15]

#a = pd.read_excel('/content/drive/MyDrive/Projects/Stat analysis extrusion research/new idea comparative study/PS-285k-2018data.xlsx')
#b = pd.read_excel('/content/drive/MyDrive/Projects/Stat analysis extrusion research/new idea comparative study/PS-285k-2018data.xlsx', sheet_name=1)
#c = pd.read_excel('/content/drive/MyDrive/Projects/Stat analysis extrusion research/new idea comparative study/PS-285k-2018data.xlsx', sheet_name=2)

#a = a.drop([0])
#b = b.drop([0])
#c = c.drop([0])

#data = [a, b, c]
#temperature = [403.15, 433.15, 443.15]
#f = f.drop([0, 1, 4])

def generalized(shear, visc, temp_list, stop, t_ref_pos):  

  def cy_func(sr, TN, params, att1, att2):
    return att1[-1][TN]*(params.params['ei'].value+(params.params['e0'].value-params.params['ei'].value)*(1+(att2[-1][TN]*params.params['l'].value*10**sr)**params.params['a'].value)**((params.params['n'].value-1)/params.params['a'].value))

  def cy_func2(sr, params):
    return params.params['ei'].value+(params.params['e0'].value-params.params['ei'].value)*(1+(params.params['l'].value*(10**sr))**params.params['a'].value)**((params.params['n'].value-1)/params.params['a'].value)

  #shear_data = [np.log10(np.array(x['Angular frequency']).astype(float)) for x in dataset]
  #viscosity_data = [np.log10(np.array(y['Complex viscosity']).astype(float)) for y in dataset]

  shear_data = shear
  viscosity_data = visc

  for i in range(len(temp_list)):
    plot(10**shear_data[i], 10**viscosity_data[i], '>', label = '%.2f K' %(temp_list[i]))

  legend(loc = 'best', prop={'size': 14}, title = 'Experimental data', title_fontsize=15)
  xlabel('$\dot{\gamma}$ ($\mathrm{s}^{-1}$)', size = 14)
  ylabel('$\eta$ (Pa$\cdot$s)', size = 14)
  xticks(fontsize=14)
  yticks(fontsize=14)
  yscale('log')
  xscale('log')
  show()

  a_T = 1

  x = shear_data[t_ref_pos]
  y = viscosity_data[t_ref_pos]

  def fitfunc_LOG_carreau(x, ei, e0, a, n, l):
      return np.log10(a_T) + np.log10(ei+(e0 - ei)*(1 + (l*a_T*10**x)**a)**((n-1)/a))
  
  gmodel = Model(fitfunc_LOG_carreau)

  params = gmodel.make_params(ei=10**min(y)/100, e0=10**max(y)*100, a=2, n = 0.5, l = 2)
  params['ei'].min = np.finfo(float).eps
  params['e0'].min = 10**max(y)
  params['a'].min = np.finfo(float).eps
  params['n'].min = np.finfo(float).eps
  params['n'].max = 1
  params['l'].min = np.finfo(float).eps

  result = gmodel.fit(y, params, x=x, method = "TNC")

  Rsq = [1 - result.redchi / np.var(result.best_fit, ddof=5)]
  aic = [result.aic]
  bic = [result.bic]

  aT1 = []
  aT2 = []

  loss = []

  for i in range(len(temp_list)):

    if i==t_ref_pos:
      aT1.append(1)
      aT2.append(1)

    if i < t_ref_pos:

      x = shear_data[i]
      y = viscosity_data[i]

      ei = result.params['ei'].value
      e0 = result.params['e0'].value
      a = result.params['a'].value
      n =  result.params['n'].value
      l = result.params['l'].value

      def fitfunc_LOG_carreau(x, a_T1, a_T2):
          return np.log10(a_T1) + np.log10(ei+(e0 - ei)*(1 + (l*a_T2*10**x)**a)**((n-1)/a))

      model = Model(fitfunc_LOG_carreau)

      ig1 = 1.1
      ig2 = 1.1
      params = model.make_params(a_T1 = ig1, a_T2 = ig2)

      params['a_T1'].min = 1
      params['a_T1'].max = 2
      params['a_T2'].min = 1
      params['a_T2'].max = 2

      result1 = model.fit(y, params, x=x, method = "TNC")
      aT1.append(result1.params['a_T1'].value)
      aT2.append(result1.params['a_T2'].value)

    if i > t_ref_pos:

      x = shear_data[i]
      y = viscosity_data[i]

      ei = result.params['ei'].value
      e0 = result.params['e0'].value
      a = result.params['a'].value
      n =  result.params['n'].value
      l = result.params['l'].value

      def fitfunc_LOG_carreau(x, a_T1, a_T2):
          return np.log10(a_T1) + np.log10(ei+(e0 - ei)*(1 + (l*a_T2*10**x)**a)**((n-1)/a))

      model = Model(fitfunc_LOG_carreau)

      ig1 = 0.1
      ig2 = 0.1

      params = model.make_params(a_T1 = ig1, a_T2 = ig2)

      params['a_T1'].min = 0
      params['a_T1'].max = 1
      params['a_T2'].min = 0
      params['a_T2'].max = 1

      result1 = model.fit(y, params, x=x, method = "TNC")
      aT1.append(result1.params['a_T1'].value)
      aT2.append(result1.params['a_T2'].value)

  global_at1 = [aT1]
  global_at2 = [aT2]

  actual  = list(np.concatenate(tuple([x for x in viscosity_data])))
  predicted = list(np.concatenate(tuple([np.log10(cy_func(x, j, result, global_at1, global_at2)) for j, x in enumerate(shear_data)])))
  mse = sk.mean_squared_error(actual, predicted)
  loss_M = math.sqrt(mse)
  loss.append(loss_M)

  global_e0 = [e0]
  global_ei = [ei]
  global_a = [a]
  global_n = [n]
  global_l = [l]

  criteria = 999

  while criteria > stop:

    shear = np.concatenate(tuple([shear_data[i] + np.log10(aT2[i]) for i in range(len(shear_data))]))
    visc = np.concatenate(tuple([viscosity_data[j] - np.log10(aT1[j]) for j in range(len(viscosity_data))]))

    x = shear
    y = visc

    def fitfunc_LOG_carreau(x, ei, e0, a, n, l):
        return np.log10(a_T) + np.log10(ei+(e0 - ei)*(1 + (l*a_T*10**x)**a)**((n-1)/a))

    gmodel = Model(fitfunc_LOG_carreau)

    params = gmodel.make_params(ei=global_ei[-1], e0=global_e0[-1], a=global_a[-1], n = global_n[-1], l = global_l[-1])
    params['ei'].min = np.finfo(float).eps
    params['e0'].min = 10**max(y)
    params['a'].min = np.finfo(float).eps
    params['n'].min = np.finfo(float).eps
    params['n'].max = 1
    params['l'].min = np.finfo(float).eps

    result = gmodel.fit(y, params, x=x, method = "TNC")

    criteria = 1 - result.redchi / np.var(result.best_fit, ddof=5)
    criteria = 1 - (1-criteria)*(len(visc)-1)/(len(visc)-6)
    Rsq.append(criteria)

    criteria = abs(Rsq[-1] - Rsq[-2])/Rsq[-2]

    aic.append(result.aic)
    bic.append(result.bic)

    global_e0.append(result.params['e0'].value)
    global_ei.append(result.params['ei'].value)
    global_a.append(result.params['a'].value)
    global_n.append(result.params['n'].value)
    global_l.append(result.params['l'].value)

    actual  = list(np.concatenate(tuple([x for x in viscosity_data])))
    predicted = list(np.concatenate(tuple([np.log10(cy_func(x, j, result, global_at1, global_at2)) for j, x in enumerate(shear_data)])))
    mse = sk.mean_squared_error(actual, predicted)
    loss_M = math.sqrt(mse)
    loss.append(loss_M)

    aT1 = []
    aT2 = []
 
    for i in range(len(temp_list)):

      if i==t_ref_pos:
        aT1.append(1)
        aT2.append(1)

      if i < t_ref_pos:

        x = shear_data[i]
        y = viscosity_data[i]

        ei = result.params['ei'].value
        e0 = result.params['e0'].value
        a = result.params['a'].value
        n =  result.params['n'].value
        l = result.params['l'].value

        def fitfunc_LOG_carreau(x, a_T1, a_T2):
            return np.log10(a_T1) + np.log10(ei+(e0 - ei)*(1 + (l*a_T2*10**x)**a)**((n-1)/a))

        model = Model(fitfunc_LOG_carreau)

        ig1 = global_at1[-1][i]
        ig2 = global_at2[-1][i]

        params = model.make_params(a_T1 = ig1, a_T2 = ig2)

        params['a_T1'].min = 1
        params['a_T1'].max = 2
        params['a_T2'].min = 1
        params['a_T2'].max = 2

        result1 = model.fit(y, params, x=x, method = "TNC")
        aT1.append(result1.params['a_T1'].value)
        aT2.append(result1.params['a_T2'].value)
      
      if i > t_ref_pos:

        x = shear_data[i]
        y = viscosity_data[i]

        ei = result.params['ei'].value
        e0 = result.params['e0'].value
        a = result.params['a'].value
        n =  result.params['n'].value
        l = result.params['l'].value

        def fitfunc_LOG_carreau(x, a_T1, a_T2):
            return np.log10(a_T1) + np.log10(ei+(e0 - ei)*(1 + (l*a_T2*10**x)**a)**((n-1)/a))

        model = Model(fitfunc_LOG_carreau)

        ig1 = global_at1[-1][i]
        ig2 = global_at2[-1][i]
        params = model.make_params(a_T1 = ig1, a_T2 = ig2)

        params['a_T1'].min = 0
        params['a_T1'].max = 1
        params['a_T2'].min = 0
        params['a_T2'].max = 1

        result1 = model.fit(y, params, x=x, method = "TNC")
        aT1.append(result1.params['a_T1'].value)
        aT2.append(result1.params['a_T2'].value)
        
    global_at1.append(aT1)
    global_at2.append(aT2)
  
  iter = [x+1 for x in range(len(Rsq))]

  fig, (ax1, ax2, ax4) = plt.subplots(3, 1, sharex=True)
  
  ax1.plot(iter, Rsq, '--k')
 
  ax1.set_ylabel('$R_{\mathrm{adj}}^2$')
 
  ax2.plot(iter, aic, '--', label = 'AIC')
  

  ax2.plot(iter, bic, '--', label = 'BIC')
  ax2.set_ylabel('Information Criterion')
  ax2.legend(loc = 'best') 

  ax4.plot(iter, loss, '-.m')
  ax4.set_xlabel('Iteration')
  ax4.set_ylabel('RMSE')

  plt.xticks(fontsize=10)
  plt.yticks(fontsize=10)

  show()

  ##########################################################################

  fig1, (ax11, ax22) = plt.subplots(2, 1, sharex=True)

  def newListRemove(element, list1): 
    return list(filter(lambda x: x != element, list1))

  #templist = newListRemove(temp_list[t_ref_pos], temp_list)

  index = [i for i in range(len(temp_list))]
  index = newListRemove(t_ref_pos, index)

  for i in index:
    ax11.plot(iter, [x[i] for x in global_at1], 'o', fillstyle='none', label = '$a_T^*$ at %.2f K' %(temp_list[i]))
  ax11.set_ylabel('Value', size = 15)
  ax11.legend(loc='best', prop={'size': 15})

  for i in index:
    ax22.plot(iter, [x[i] for x in global_at2], 'o', fillstyle='none', label = '$a_T^*$ at %.2f K' %(temp_list[i]))
  ax22.set_xlabel('Iteration', size = 15)
  ax22.set_ylabel('Value', size = 15)
  ax22.legend(loc='best', prop={'size': 15})
  show()

  ##########################################################################

  plot(global_ei, 's-')
  ylabel("$\eta_{\infty}$", size = 15)
  xlabel("Iteration", size = 15)
  show()

  plot(global_e0, 's-')
  ylabel("$\eta_0$", size = 15)
  xlabel("Iteration", size = 15)
  show()

  plot(global_a, 's-')
  ylabel("$a$", size = 15)
  xlabel("Iteration", size = 15)
  show()

  plot(global_n, 's-')
  ylabel("$n$", size = 15)
  xlabel("Iteration", size = 15)
  show()

  plot(global_l, 's-')
  ylabel("$\lambda$", size = 15)
  xlabel("Iteration", size = 15)
  show()

  shearspace = np.linspace(min(10**shear)*0.95, max(10**shear)*1.02, 5000000)

  for i in range(len(temp_list)):
    plot(10**shear_data[i], 10**viscosity_data[i], 'o', fillstyle='none', label = '%.2f K' %(temp_list[i]))

  for i in range(len(temp_list)):
    plot(10**np.linspace(min(shear_data[i])*0.95, max(shear_data[i])*1.02, 5000000), cy_func(np.linspace(min(shear_data[i])*0.95, max(shear_data[i])*1.02, 5000000), i, result, global_at1, global_at2), linewidth=1.5, label = ' ')

  xlabel("$\dot{\gamma}$ ($s^{-1}$)", size = 14)
  ylabel("$\eta$ (Pa$\cdot$s)", size = 14)
  xticks(fontsize=14)
  yticks(fontsize=14)
  yscale('log')
  xscale('log')
  legend(loc='best', prop={'size': 14}, ncol=2, title = 'Data          Model', title_fontsize=15)
  show()

  for i in range(len(temp_list)):
    plot(10**shear_data[i]*global_at2[-1][i], 10**viscosity_data[i]/global_at1[-1][i], 'o', fillstyle='none', label = '%.2f K' %(temp_list[i]))
  plot(shearspace, cy_func2(np.log10(shearspace), result), linewidth=1.5, color = 'k', label = 'Model')
  xlabel("$\dot{\gamma}a_{T}$", size = 14)
  ylabel("$\eta/a_{T}^*$", size = 14)
  xticks(fontsize=14)
  yticks(fontsize=14)
  yscale('log')
  xscale('log')
  legend(loc='best', prop={'size': 11}, title_fontsize=12)
  show()
  
  T = []
  for i in range(len(temp_list)):
    T.append(np.log10(cy_func(shear_data[i], i, result, global_at1, global_at2)))

  return [global_ei, global_e0, global_a, global_n, global_l], Rsq, aic, bic, global_at1, global_at2, loss[-1], T

pars1, rsq1, aic1, bic1, at11, at21, loss1, predicted1 = generalized(shear, visc, temperature, 0.000000001, 2)

"""#END OF CODE"""