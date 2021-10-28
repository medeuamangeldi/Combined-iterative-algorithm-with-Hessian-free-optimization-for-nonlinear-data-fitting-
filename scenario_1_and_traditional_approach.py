import pandas as pd
import numpy as np
from pylab import *
from scipy.optimize import curve_fit
!pip install lmfit
!pip install numdifftools
import numdifftools
from lmfit import Model
import sklearn.metrics as sk

from matplotlib.ticker import LinearLocator

import warnings
warnings.filterwarnings("ignore")

"""#Upload and organize the data"""

rcParams["figure.figsize"] = (8,6)
matplotlib.rcParams["figure.dpi"] = 300
style.use('seaborn-ticks')
rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'
matplotlib.rcParams['lines.markersize'] = 10

a = pd.read_excel('/content/drive/MyDrive/Projects/Stat analysis extrusion research/new idea comparative study/PS-285k-2018data.xlsx')
b = pd.read_excel('/content/drive/MyDrive/Projects/Stat analysis extrusion research/new idea comparative study/PS-285k-2018data.xlsx', sheet_name=1)
c = pd.read_excel('/content/drive/MyDrive/Projects/Stat analysis extrusion research/new idea comparative study/PS-285k-2018data.xlsx', sheet_name=2)
d = pd.read_excel('/content/drive/MyDrive/Projects/Stat analysis extrusion research/new idea comparative study/PS-285k-2018data.xlsx', sheet_name=3)

a = a.drop([0])
b = b.drop([0])
c = c.drop([0])
d = d.drop([0])

#f = f.drop([0, 1, 4])

shear130, visc130 = np.log10(np.array(a['Angular frequency']).astype(float)), np.log10(np.array(a['Complex viscosity']).astype(float))
shear160, visc160 = np.log10(np.array(b['Angular frequency']).astype(float)), np.log10(np.array(b['Complex viscosity']).astype(float))
shear170, visc170 = np.log10(np.array(c['Angular frequency']).astype(float)), np.log10(np.array(c['Complex viscosity']).astype(float))

plot(10**shear130, 10**visc130, '^', label = '403.15 K', markersize=12)
plot(10**shear160, 10**visc160, 'o', label = '433.15 K',markersize=12)
plot(10**shear170, 10**visc170, 's', label = '443.15 K', markersize=12)
legend(loc = 'best', prop={'size': 16}, title = 'Experimental data', title_fontsize=16)
xlabel('$\dot{\gamma}$ ($\mathrm{s}^{-1}$)', size = 17)
ylabel('$\eta$ (Pa$\cdot$s)', size = 17)
xticks(fontsize=16)
yticks(fontsize=16)
yscale('log')
xscale('log')
show()

"""##Define the function that performs the IQR method and remove the outliers"""

shear = (shear130, shear160, shear170)
visc = (visc130, visc160, visc170)

def iterative_approach(shear_rate, viscosity, stop, TREF):

  def cy_func(sr, TN, params, att1, att2):
    return att1[-1][TN]*(params.params['ei'].value+(params.params['e0'].value-params.params['ei'].value)*(1+(att2[-1][TN]*params.params['l'].value*10**sr)**params.params['a'].value)**((params.params['n'].value-1)/params.params['a'].value))

  def cy_func2(sr, params):
    return params.params['ei'].value+(params.params['e0'].value-params.params['ei'].value)*(1+(params.params['l'].value*(10**sr))**params.params['a'].value)**((params.params['n'].value-1)/params.params['a'].value)

  def cy_func1(sr, TN, params, att1, att2):
      return att1[-1][TN]*(params.params['ei'].value+(params.params['e0'].value-params.params['ei'].value)*(1+(att2[-1][TN]*params.params['l'].value*10**sr)**params.params['a'].value)**((params.params['n'].value-1)/params.params['a'].value))

  temp = [i for i in range(3)]

  temp.remove(int(TREF-1))

  a_T = 1
  combined_visc = (viscosity[0] - np.log10(1), viscosity[1] - np.log10(0.00352186840222), viscosity[2] - np.log10(0.00102299342382))
  combined_shear = (shear_rate[0] + np.log10(1), shear_rate[1] + np.log10(0.00355286), shear_rate[2] + np.log10(0.00108961))

  visc = np.concatenate(combined_visc)
  shear = np.concatenate(combined_shear)

  x = shear
  y = visc

  #x = shear_rate[TREF-1]
  #y = viscosity[TREF-1]

  def fitfunc_LOG_carreau(x, ei, e0, a, n, l):
      return np.log10(a_T) + np.log10(ei+(e0 - ei)*(1 + (l*a_T*10**x)**a)**((n-1)/a))
  
  gmodel = Model(fitfunc_LOG_carreau)

  params = gmodel.make_params(ei=10**min(y)/5, e0=10**max(y)*1.5, a=2, n = 0.5, l = 2)
  params['ei'].min = np.finfo(float).eps
  #params['ei'].max = 10**min(y)
  params['e0'].min = 10**max(y)
  params['a'].min = np.finfo(float).eps
  params['n'].min = np.finfo(float).eps
  params['n'].max = 1
  params['l'].min = np.finfo(float).eps

  result = gmodel.fit(y, params, x=x, method = "TNC")

  Rsq = [1 - result.redchi / np.var(result.best_fit, ddof=5)]
  aic = [result.aic]
  bic = [result.bic]

  loss = []

  aT1 = []
  aT2 = []

  for i in temp:
    if i==int(TREF):
      aT1.append(1)
      aT2.append(1)

    x = shear_rate[i]
    y = viscosity[i]

    ei = result.params['ei'].value
    e0 = result.params['e0'].value
    a = result.params['a'].value
    n =  result.params['n'].value
    l = result.params['l'].value

    def fitfunc_LOG_carreau(x, a_T1, a_T2):
        return np.log10(a_T1) + np.log10(ei+(e0 - ei)*(1 + (l*a_T2*10**x)**a)**((n-1)/a))

    model = Model(fitfunc_LOG_carreau)
    if i != 2:
      ig1 = 0.00352186840222
      ig2 = 0.00355286
      params = model.make_params(a_T1 = ig1, a_T2 = ig2)

      params['a_T1'].min = 0
      params['a_T1'].max = 1
      params['a_T2'].min = 0
      params['a_T2'].max = 1

      result1 = model.fit(y, params, x=x, method = "TNC")
      aT1.append(result1.params['a_T1'].value)
      aT2.append(result1.params['a_T2'].value)
    else:
      ig1 = 0.00102299342382
      ig2 = 0.00108961
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

  actual  = list(np.concatenate((visc130, visc160, visc170)))
  predicted = list(np.concatenate((np.log10(cy_func(shear130, 0, result, global_at1, global_at2)), np.log10(cy_func(shear160, 1, result, global_at1, global_at2)), np.log10(cy_func(shear170, 2, result, global_at1, global_at2)))))
  mse = sk.mean_squared_error(actual, predicted)
  loss_M = math.sqrt(mse)
  loss.append(loss_M)

  global_e0 = [e0]
  global_ei = [ei]
  global_a = [a]
  global_n = [n]
  global_l = [l]

  criteria = 1

  while criteria > stop:

    combined_visc = (viscosity[0] - np.log10(aT1[0]), viscosity[1] - np.log10(aT1[1]), viscosity[2] - np.log10(aT1[2]))
    combined_shear = (shear_rate[0] + np.log10(aT2[0]), shear_rate[1] + np.log10(aT2[1]), shear_rate[2] + np.log10(aT2[2]))

    visc = np.concatenate(combined_visc)
    shear = np.concatenate(combined_shear)

    x = shear
    y = visc

    def fitfunc_LOG_carreau(x, ei, e0, a, n, l):
        return np.log10(a_T) + np.log10(ei+(e0 - ei)*(1 + (l*a_T*10**x)**a)**((n-1)/a))

    gmodel = Model(fitfunc_LOG_carreau)

    params = gmodel.make_params(ei = global_ei[-1], e0 = global_e0[-1], a = global_a[-1], n = global_n[-1], l = global_l[-1])
    params['ei'].min = np.finfo(float).eps
    #params['ei'].max = 10**min(y)
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

    actual  = list(np.concatenate((visc130, visc160, visc170)))
    predicted = list(np.concatenate((np.log10(cy_func(shear130, 0, result, global_at1, global_at2)), np.log10(cy_func(shear160, 1, result, global_at1, global_at2)), np.log10(cy_func(shear170, 2, result, global_at1, global_at2)))))
    mse = sk.mean_squared_error(actual, predicted)
    loss_M = math.sqrt(mse)
    loss.append(loss_M)

    aT1 = []
    aT2 = []

    for i in temp:
      if i==int(TREF):
        aT1.append(1)
        aT2.append(1)
      x = shear_rate[i]
      y = viscosity[i]

      ei = result.params['ei'].value
      e0 = result.params['e0'].value
      a = result.params['a'].value
      n =  result.params['n'].value
      l = result.params['l'].value

      def fitfunc_LOG_carreau(x, a_T1, a_T2):
          return np.log10(a_T1) + np.log10(ei+(e0 - ei)*(1 + (l*a_T2*10**x)**a)**((n-1)/a))

      model = Model(fitfunc_LOG_carreau)

      if i != 2:
        ig1 = global_at1[-1][1]
        ig2 = global_at2[-1][1]
        params = model.make_params(a_T1 = ig1, a_T2 = ig2)

        params['a_T1'].min = 0
        params['a_T1'].max = 1
        params['a_T2'].min = 0
        params['a_T2'].max = 1

        result1 = model.fit(y, params, x=x, method = "TNC")
        aT1.append(result1.params['a_T1'].value)
        aT2.append(result1.params['a_T2'].value)
      else:
        ig1 = global_at1[-1][2]
        ig2 = global_at2[-1][2]
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

  ax11.plot(iter, [x[1] for x in global_at1], 'o-', fillstyle = 'none', label = '$a_T^*$ at 433.15 K')
  ax11.plot(iter, [x[1] for x in global_at2], 'o-', fillstyle = 'none', label = '$a_T^*$ at 443.15 K')
  ax11.set_ylabel('Value', size = 15)
  ax11.legend(loc='best', prop={'size': 15})
  
  ax22.plot(iter, [x[2] for x in global_at1], '^-', fillstyle = 'none', label = '$a_T$ at 433.15 K')
  ax22.plot(iter, [x[2] for x in global_at2], '^-', fillstyle = 'none', label = '$a_T$ at 443.15 K')
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


  shearspace = np.linspace(10**min(x), 100, 2000)
  shearspace1 = np.linspace(10**(-4), 100, 1000000)
  shearspace2 = np.linspace(-4.5, 2, 1000)
  shearspace3 = np.linspace(-4.6, 2, 1000)

  shearspace11 = np.linspace(min(shear130), max(shear130), 1000000)
  shearspace22 = np.linspace(min(shear160), max(shear160), 1000000)
  shearspace33 = np.linspace(min(shear170), max(shear170), 1000000)

  plot(10**shear130, 10**visc130, '^', label = '403.15 K')
  plot(10**shear160, 10**visc160, 'o', label = '433.15 K')
  plot(10**shear170, 10**visc170, 's', label = '443.15 K')

  plot(10**shearspace11, cy_func(shearspace11, 0, result, global_at1, global_at2), label = ' ')
  plot(10**shearspace22, cy_func(shearspace22, 1, result, global_at1, global_at2), label = ' ')
  plot(10**shearspace33, cy_func(shearspace33, 2, result, global_at1, global_at2), label = ' ')

  xlabel("$\dot{\gamma}$ ($s^{-1}$)", size = 14)
  ylabel("$\eta$ (Pa$\cdot$s)", size = 14)
  xticks(fontsize=14)
  yticks(fontsize=14)
  yscale('log')
  xscale('log')
  legend(loc='best', prop={'size': 14}, ncol=2, title = 'Data               Model', title_fontsize=15)
  show()

  plot(10**shear130*global_at2[-1][0], 10**visc130/global_at1[-1][0], '^', label = '403.15 K')
  plot(10**shear160*global_at2[-1][1], 10**visc160/global_at1[-1][1], 'o', label = '433.15 K')
  plot(10**shear170*global_at2[-1][2], 10**visc170/global_at1[-1][2], 's', label = '443.15 K')
  plot(10**shearspace2, cy_func2(shearspace2, result), label = 'Model')
  xlabel("$\dot{\gamma}a_{T}$", size = 14)
  ylabel("$\eta/a_{T}^*$", size = 14)
  xticks(fontsize=14)
  yticks(fontsize=14)
  yscale('log')
  xscale('log')
  legend(loc='best', prop={'size': 14}, title_fontsize=15)
  show()
  
  T130 = np.array([np.log10(cy_func(x, 0, result, global_at1, global_at2)) for x in shear130])
  T160 = np.array([np.log10(cy_func(x, 1, result, global_at1, global_at2)) for x in shear160])
  T170 = np.array([np.log10(cy_func(x, 2, result, global_at1, global_at2)) for x in shear170])

  return [global_ei, global_e0, global_a, global_n, global_l], Rsq, aic, bic, global_at1, global_at2, loss[-1], [T130, T160, T170]

pars, rsq, aic, bic, at1, at2, loss, predicted = iterative_approach(shear, visc, 1E-8, 1)

#her master curve fitting for Huang's data

combined_visc = (visc130 - np.log10(1), visc160 - np.log10(0.00352186840222), visc170 - np.log10(0.00102299342382))
combined_shear = (shear130 + np.log10(1), shear160 + np.log10(0.00355286), shear170 + np.log10(0.00108961))

visc = np.concatenate(combined_visc)
shear = np.concatenate(combined_shear)

x = shear
y = visc

def fitfunc_LOG_carreau(x, ei, e0, a, n, l):
    return np.log10(ei+(e0 - ei)*(1 + (l*10**x)**a)**((n-1)/a))

gmodel = Model(fitfunc_LOG_carreau)

params = gmodel.make_params(ei=10**min(y)/5, e0=10**max(y)*1.5, a=2, n = 0.5, l = 2)
params['ei'].min = np.finfo(float).eps
#params['ei'].max = 10**min(y)
params['e0'].min = 10**max(y)
params['a'].min = np.finfo(float).eps
params['n'].min = np.finfo(float).eps
params['n'].max = 1
params['l'].min = np.finfo(float).eps

result = gmodel.fit(y, params, x=x, method = "TNC")


plt.plot(10**shear130, 10**visc130, '^', label = '403.15 K')
plt.plot(10**shear160*0.00355286, 10**visc160/0.00352186840222, 'o', label = '433.15 K')
plt.plot(10**shear170*0.00108961, 10**visc170/0.00102299342382, 's', label = '443.15 K')
shearspace4 = np.linspace(-4.6, 2, 10000000)

eif = result.params['ei'].value
e0f = result.params['e0'].value
af = result.params['a'].value
nf =  result.params['n'].value
lf = result.params['l'].value

def cy_func3(sr, params):
    return params.params['ei'].value+(params.params['e0'].value-params.params['ei'].value)*(1+(params.params['l'].value*(10**sr))**params.params['a'].value)**((params.params['n'].value-1)/params.params['a'].value)

plt.plot(10**shearspace4, cy_func3(shearspace4, result), label = 'Model')

criteria = 1 - result.redchi / np.var(result.best_fit, ddof=5)
criteria = 1 - (1-criteria)*(len(visc)-1)/(len(visc)-6)

print(criteria, result.aic, result.bic)
print(eif, e0f, af, nf, lf)

plt.xlabel("$\dot{\gamma}a_T$", size = 14)
plt.ylabel("$\eta/a_T^*$", size = 14)
plt.legend(loc='best', prop={'size': 14})
xticks(fontsize=14)
yticks(fontsize=14)
yscale('log')
xscale('log')
#plt.grid()
plt.show()

plt.plot(10**shear130, 10**visc130, '^', label = '403.15 K')
plt.plot(10**shear160, 10**visc160, 'o', label = '433.15 K')
plt.plot(10**shear170, 10**visc170, 's', label = '443.15 K')

shearspace11 = np.linspace(min(shear130), max(shear130), 1000000)
shearspace22 = np.linspace(min(shear160), max(shear160), 1000000)
shearspace33 = np.linspace(min(shear170), max(shear170), 1000000)
def cy_func4(sr, TN, params, att1, att2):
    return att1[TN]*(params.params['ei'].value+(params.params['e0'].value-params.params['ei'].value)*(1+(att2[TN]*params.params['l'].value*10**sr)**params.params['a'].value)**((params.params['n'].value-1)/params.params['a'].value))

plt.plot(10**shearspace11, cy_func4(shearspace11, 0, result, [1, 0.00355286, 0.00108961], [1, 0.00352186840222, 0.00102299342382]), label = ' ')
plt.plot(10**shearspace22, cy_func4(shearspace22, 1, result, [1, 0.00355286, 0.00108961], [1, 0.00352186840222, 0.00102299342382]),label = ' ')
plt.plot(10**shearspace33, cy_func4(shearspace33, 2, result, [1, 0.00355286, 0.00108961], [1, 0.00352186840222, 0.00102299342382]), label = ' ')

T130 = np.array([np.log10(cy_func4(x, 0, result, [1, 0.00355286, 0.00108961], [1, 0.00352186840222, 0.00102299342382])) for x in shear130])
T160 = np.array([np.log10(cy_func4(x, 1, result, [1, 0.00355286, 0.00108961], [1, 0.00352186840222, 0.00102299342382])) for x in shear160])
T170 = np.array([np.log10(cy_func4(x, 2, result, [1, 0.00355286, 0.00108961], [1, 0.00352186840222, 0.00102299342382])) for x in shear170])


actual  = list(np.concatenate((visc130, visc160, visc170)))
predicted = list(np.concatenate((np.log10(cy_func4(shear130, 0, result, [1, 0.00355286, 0.00108961], [1, 0.00352186840222, 0.00102299342382])), np.log10(cy_func4(shear160, 1, result, [1, 0.00355286, 0.00108961], [1, 0.00352186840222, 0.00102299342382])), np.log10(cy_func4(shear170, 2, result, [1, 0.00355286, 0.00108961], [1, 0.00352186840222, 0.00102299342382])))))
mse = sk.mean_squared_error(actual, predicted)
lossQ= math.sqrt(mse)


plt.xlabel("$\dot{\gamma}$ ($\mathrm{s}^{-1}$)", size = 14)
plt.ylabel("$\eta$ (Pa$\cdot$s)", size = 14)
plt.legend(loc='best', prop={'size': 14}, ncol = 2, title_fontsize=15, title = 'Data               Model')
xticks(fontsize=14)
yticks(fontsize=14)
yscale('log')
xscale('log')
plt.show()

plot(10**shear130*1, 10**visc130, 's', label = '403.15 K')
plot(10**shear160*0.00355286, 10**visc160/0.00352186840222, 's', label = '433.15 K')
plot(10**shear170*0.00108961, 10**visc170/0.00102299342382, 's', label = '443.15 K')
xlabel('$\dot{\gamma}a_T$', size = 14)
ylabel('$\eta/(a_Tb_T)$', size = 14)
legend(prop={'size': 14}, title = 'Experimental data')
xticks(fontsize=14)
yticks(fontsize=14)
yscale('log')
xscale('log')
show()

at11 = [1, 0.00352186840222, 0.00102299342382]
at22 = [1, 0.00355286, 0.00108961]

T = [403.15, 433.15, 443.15]

plt.plot(T, np.log10(at11), 'o', label = 'Traditional method')
plt.plot(T, np.log10(at1[-1]), 'o', label = 'Scenario I')
plt.xlabel("  Temperature (K)")
plt.ylabel("Shift factors in the Carreau-Yasuda model")
plt.legend(loc='best', ncol = 2, prop={'size': 6}, title = '$a_T^*$', title_fontsize=6)
plt.show()

plt.plot(T, np.log10(at22), 'o', label = 'Traditional method')
plt.plot(T, np.log10(at2[-1]), 'o', label = 'Scenario I')
plt.xlabel("  Temperature (K)")
plt.ylabel("Shift factors in the Carreau-Yasuda model")
plt.legend(loc='best', ncol = 2, prop={'size': 6}, title = '$a_T$', title_fontsize=6)
plt.show()

def funcs1(x, a, b):
  return -a*(x-403.15)/(b+(x-403.15))

def funcs(x, a, b):
  return -a*(x-403.15)/(b+(x-403.15))

polyline = np.linspace(403.15, 443.15, 1000)

popt1, pcov1 = curve_fit(funcs, T, np.log10(at1[-1]), method = 'lm')
popt2, pcov2 = curve_fit(funcs1, T, np.log10(at2[-1]), method = 'lm')
popt11, pcov11 = curve_fit(funcs, T, np.log10(at11), method = 'lm')
popt22, pcov22 = curve_fit(funcs1, T, np.log10(at22), method = 'lm')

plt.plot(T, at11, 'p', label = '$a_T^*$ Trad')
plt.plot(T, at22, 's', label = '$a_T$ Trad')
plt.plot(T, at1[-1], '^', label = '$a_T^*$ Sc I')
plt.plot(T, at2[-1], 'o', label = '$a_T$ Sc I')

plt.plot(polyline, 10**funcs(polyline, *popt11),  label = ' ')
plt.plot(polyline, 10**funcs1(polyline, *popt22), label = ' ')
plt.plot(polyline, 10**funcs(polyline, *popt1), label = ' ')
plt.plot(polyline, 10**funcs1(polyline, *popt2), label = ' ')



plt.xlabel("Temperature (K)", size = 14)
plt.ylabel("Value", size = 14)
xticks(fontsize=14)
yticks(fontsize=14)
yscale('log')
plt.legend(loc='best', ncol = 2, prop={'size': 6}, title = 'Shift factors   WLF model', title_fontsize=6)
plt.show()

from matplotlib import ticker, cm
from matplotlib.colors import LogNorm
rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'
import matplotlib.colors as colors

def cy_optimized(shear, t,  p, pp1, pp2):
  return 10**funcs(t, *pp1)*(p[0][-1]+(p[1][-1]-p[0][-1])*(1+abs(p[4][-1]*(10**funcs1(t, *pp2))*shear)**p[2][-1])**((p[3][-1]-1)/p[2][-1]))

NN = 3000

t = np.linspace(403.15, 443.15, NN)
sr = np.linspace(min(10**shear130), 100, NN)

sr, t = np.meshgrid(sr, t)
v1 = cy_optimized(sr, t, pars, popt1, popt2)
v2 = cy_optimized(sr, t, pars, popt11, popt22)

fig1 = plt.figure(figsize=(6,4))
ax1 = fig1.add_subplot(221)  
cs1 = ax1.contourf(np.log10(sr), t, np.log10(v1), cmap='jet', levels = 100)

#norm=colors.LogNorm(vmin=v1.min(), vmax=v2.max())

cbar1 = plt.colorbar(cs1, extend='max', label = '$\eta$ (Pa$\cdot$s)')
plt.title('Scenario I')
ax1.set_ylabel('T (K)')
ax1.set_xlabel('$\dot{\gamma}$ ($\mathrm{s}^{-1}$)')
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# ax.plot_surface(np.log10(sr), t, np.log10(v1), rstride=5, cs1tride=5, cmap=cm.autumn, alpha=0.8, linewidth=0.3, antialiased=True, label = 'hello')
# ax.plot_surface(np.log10(sr), t, np.log10(v2), rstride=5, cs1tride=5, cmap=cm.autumn, alpha=0.8, linewidth=0.3, antialiased=True)

# ax.scatter3D(shear130, [T[0]]*len(shear130), visc130, label = '403.15 K')
# ax.scatter3D(shear160, [T[1]]*len(shear160), visc160, label = '433.15 K')
# ax.scatter3D(shear170, [T[2]]*len(shear170), visc170, label = '443.15 K')

# ax.set_xlabel('$\dot{\gamma}$')
# ax.set_ylabel('T')
# ax.set_zlabel('$\eta$', rotation = 0)

# ax.legend(title = 'Data', prop={'size': 6}, title_fontsize=6)

# ax.view_init(15,45)

# plt.show()
ax2 = fig1.add_subplot(222)
cs2 = ax2.contourf(np.log10(sr), t, np.log10(v2), cmap='jet', levels = 100)
cbar2 = plt.colorbar(cs2, extend='max', label = '$\eta$ (Pa$\cdot$s)')
plt.title('Traditional Method')
ax2.set_ylabel('T (K)')
ax2.set_xlabel('$\dot{\gamma}$ ($\mathrm{s}^{-1}$)')

ax3 = fig1.add_subplot(223)
cs3 = ax3.contourf(np.log10(sr), t, abs(np.log10(v1)-np.log10(v2)), cmap='plasma', levels = 100)
cbar3 = plt.colorbar(cs3, extend='max', label = '$|\eta_{Scen I} - \eta_{Trad}|$')
plt.title('Absolute Error')
ax3.set_ylabel('T (K)')
ax3.set_xlabel('$\dot{\gamma}$ ($\mathrm{s}^{-1}$)')

ax4 = fig1.add_subplot(224)
cs4 = ax4.contourf(np.log10(sr), t, abs((np.log10(v1)-np.log10(v2))/((np.log10(v1)+np.log10(v2))/2)), cmap='cool', levels = 100)
cbar4 = plt.colorbar(cs4, extend='max', label = '$|(\eta_{Scen I} - \eta_{Trad})/\eta_{Scen I}|$')
plt.title('Absolute Percent Error')
ax4.set_ylabel('T (K)')
ax4.set_xlabel('$\dot{\gamma}$ ($\mathrm{s}^{-1}$)')

plt.tight_layout()
plt.show()

"""#END OF CODE"""