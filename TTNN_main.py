import pickle
from pymatgen.ext import matproj
from matminer.featurizers.structure import *
from matminer.featurizers.base import *
from matminer.featurizers.composition import *
from matminer.featurizers.site import *
from matminer.featurizers.structure import SiteStatsFingerprint
from pymatgen.analysis.local_env import VoronoiNN
import warnings 
warnings.filterwarnings("ignore")
if ijp==0:
  import pickle
  from pymatgen.ext import matproj
  from matminer.featurizers.structure import *
  from matminer.featurizers.base import *
  from matminer.featurizers.composition import *
  from matminer.featurizers.site import *
  from matminer.featurizers.structure import SiteStatsFingerprint
  from pymatgen.analysis.local_env import VoronoiNN
  fp = open ('features_union2_150', 'rb')
  features_li = pickle.load(fp)
  fp.close()
  print(len(features_li))
  features_li

  from keras.models import load_model
  model = load_model('TTNN.h5',compile=False)
  model.compile(loss = 'mae', metrics=['mae'],optimizer=keras.optimizers.Adam(lr=0.006))
  
  mae_H = np.load('MAE_phase_stab.npy')
  ijp=1

class key_gen:
  
  def __init__(self):
    import random
    self.keys = []
    
  def add_entropy(self,step=10):
    Temp = list(range(5,801,step))
    if 300 not in Temp:
      Temp.append(300)
      Temp.sort()
    entr = []
    for T in Temp:
      entr.append("S_"+str(T))
    self.keys.append(entr)
      
  def add_helmoltz(self,step=10):
    Temp = range(5,801,step)
    hm = []
    for T in Temp:
      hm.append("H_"+str(T))
    self.keys.append(hm)
      
  def add_heatCapacity(self,step=10):
    Temp = range(5,801,step)
    hc = []
    for T in Temp:
      hc.append("C_v_"+str(T))
    self.keys.append(hc)
      
  def add_energy(self,step=10):
    Temp = range(5,801,step)
    en = []
    for T in Temp:
      en.append("U_"+str(T))
    self.keys.append(en)
      
  def add_all(self,step=10):
    self.add_entropy(step=step)
    self.add_helmoltz(step=step)
    self.add_heatCapacity(step=step)
    self.add_energy(step=step)
      
  def shuffle(self):
    #random.shuffle(keys)
    pass
  
  def get_index(self,str):
    keys_flatten = [k for a in self.keys for k in a]
    return keys_flatten.index(str)
  
  def get_keys(self):
    keys_flatten = [k for a in self.keys for k in a]
    return keys_flatten
  
  def get_keys_hierarchy(self):
    return self.keys
  
  def get_keys_idx(self):
    keys_flatten = [k for a in self.keys for k in a]
    return list(enumerate(keys_flatten))
  
def get_data(keys='All'):
  
  keys_all = []
  Temp = range(5,801,5)
  for T in Temp:
        key = ["S_"+str(T),"C_v_"+str(T),"H_"+str(T),"U_"+str(T)]
        keys_all+=key
  if keys == 'All':
    keys = keys_all
  
  df_f = pd.read_pickle('df_phonon_data')

  a = {}
  for x in df_f.columns:
    if 'ChemEnvSiteFingerprint|GaussianSymmFunc' in x:
        a[x] = 'GaussianSymmFunc|'+x[39:]
  df_f = df_f.rename(a,axis=1)
  
  df_Y = df_f[keys]
  df_X = df_f.drop(keys_all,axis=1)
  
  #df_X = (df_X-df_X.min())/(df_X.max()-df_X.min())
  #df_Y = (df_Y-df_Y.min())/(df_Y.max()-df_Y.min())
  
  return (df_X,df_Y)

cross_mi = pd.read_pickle('Features_cross')
S300_mi = pd.read_pickle('Features_S300')
a = []
for x in cross_mi.index:
    if x not in S300_mi.index:
        a.append(x)
        
cross_mi = cross_mi.drop(a,axis=0).drop(a,axis=1)

def get_features(n,file):
    p=4
    c = 1e-5
    S300_mi = pd.read_pickle(file)
    score = cross_mi.copy()
    for i in score.index:
        row = score.loc[i,:]
        score.loc[i,:] = S300_mi[i] /(row**p+c)

    first_feature = S300_mi.nlargest(1).index[0]
    score = score.drop(first_feature,axis=0)
    feature_set = [first_feature]

    for _ in range(n-1):
        next_feature = score[feature_set].min(axis=1).idxmax(axis=0)
        feature_set.append(next_feature)
        score = score.drop(next_feature,axis=0)
    #print(feature_set)
    return feature_set
  
def get_features_dyn(n_feat,file):

    S300_mi = pd.read_pickle(file)
    
    first_feature = S300_mi.nlargest(1).index[0]
    feature_set = [first_feature]

    for n in range(n_feat):
        p = 4.5-(n**0.4)*0.4
        c = 0.000001*n**3
        if c > 100000:
            c=100000
        if p < 0.1:
            p=0.1
            
        #0: p = 6-2*math.log10(n+1)
        #0: c = 0.0001*(2**(n/10))
        
        #1: p = 6-2*math.log10(n+1)
        #1: c = 0.0001*(4**(n/10))
        
        #print((n,p,c))
        score = cross_mi.copy()
        score = score.drop(feature_set,axis=0)
        score = score[feature_set]
        
        for i in score.index:
            row = score.loc[i,:]
            score.loc[i,:] = S300_mi[i] /(row**p+c)
            
        next_feature = score.min(axis=1).idxmax(axis=0)
        feature_set.append(next_feature)
        
    #print(feature_set)
    return feature_set
  
def get_features_union(n):
  
  keyg = key_gen()
  keyg.add_all(step=195)
  keys = keyg.get_keys()
  print(len(keys))
  features = set()
  for k in keys:
    print(k)
    features_k = get_features_dyn(n,'Features_'+k)
    features = features.union(set(features_k))

  return list(features)

def get_features_union2(n):
  
  keyg = key_gen()
  keyg.add_all(step=50)
  keys = keyg.get_keys()
  print(len(keys))
  features = set()
  for k in keys:
    print(k)
    features_k = get_features_dyn(n,'MIS/Features_'+k)
    features = features.union(set(features_k))

  return list(features)


#df_fscale = pd.read_pickle('df_phonon_data')
#S300scale = (df_fscale['S_300'].max()-df_fscale['S_300'].min())
# equals 66.02420679123546
def process_history(hist,keys,plot=False,treat_mape=False):
  mn_range = 30
  
  df_f = pd.read_pickle('df_phonon_data')
  
  if treat_mape:
    df = pd.DataFrame([],index=keys,columns = ['mae','mape'],dtype='float')
    for k in keys:
      mae_l = hist.history['val_{}_mean_absolute_error'.format(k)]
      mape_l = hist.history['val_{}_mean_absolute_percentage_error'.format(k)]
      score_mae = 0.5*(np.sort(mae_l)[:mn_range].mean()) + 0.5*(np.array(mae_l[-mn_range:]).mean())
      score_mape = 0.5*(np.sort(mape_l)[:mn_range].mean()) + 0.5*(np.array(mape_l[-mn_range:]).mean())
      df.loc[k,'mae'] = score_mae *(df_f[k].max()-df_f[k].min())
      df.loc[k,'mape'] = score_mape
      
  else:
    df = pd.DataFrame([],index=keys,columns = ['mae'],dtype='float')
    df_scaled = pd.DataFrame([],index=keys,columns = ['mae'],dtype='float')
    for k in keys:
      mae_l = hist.history['val_{}_mean_absolute_error'.format(k)]
      score_mae = 0.5*(np.sort(mae_l)[:mn_range].mean()) + 0.5*(np.array(mae_l[-mn_range:]).mean())
      df.loc[k,'mae'] = score_mae *(df_f[k].max()-df_f[k].min())
      df_scaled.loc[k,'mae'] = score_mae
    vl = hist.history['val_loss']
    validation_loss = 0.5*(np.sort(vl)[:mn_range].mean()) + 0.5*(np.array(vl[-mn_range:]).mean())
    
  
  if plot:
    fig,ax = plt.subplots()

    ax.plot(hist.history['U_605_mean_absolute_error'.format(k)],label='train')
    ax.plot(hist.history['val_U_605_mean_absolute_error'.format(k)],label='valid')
    ax.legend()


    fig,ax = plt.subplots()

    ax.plot(hist.history['S_300_mean_absolute_error'],label='train')
    ax.plot(hist.history['val_S_300_mean_absolute_error'],label='valid')
    ax.legend()
    
  return df, df_scaled, validation_loss

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth
 
def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    """
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    import numpy as np
    from math import factorial
    
    window_size = np.abs(np.int(window_size))
    order = np.abs(np.int(order))
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')
    
def featurize_struct(struct):
    df = pd.DataFrame({'structure':struct,'composition':[x.composition for x in struct]})
    df_tot = df

    cm = CoulombMatrix()
    cm.fit(df_tot["structure"])
    scm = SineCoulombMatrix()
    scm.fit(df_tot["structure"])
    bf = BondFractions()
    bf.fit(df_tot["structure"])
    featurizer = MultipleFeaturizer([DensityFeatures(),
                                     GlobalSymmetryFeatures(),
                                     #Dimensionality(),
                                     RadialDistributionFunction(),
                                     #prdf,
                                     #ElectronicRadialDistributionFunction(),
                                     cm,
                                     scm,
                                     #OrbitalFieldMatrix(),
                                     #MinimumRelativeDistances(),
                                     #SiteStatsFingerprint(),
                                     EwaldEnergy(),
                                     bf, # add bond type name, here no distinction                                 
                                     #bob,
                                     StructuralHeterogeneity(),
                                     MaximumPackingEfficiency(),
                                     ChemicalOrdering(),
                                     XRDPowderPattern()
                                    ])


    featurizer.featurize_dataframe(df,"structure",multiindex=True,ignore_errors=True)
    df.columns = df.columns.map('|'.join).str.strip('|')
    
    dist = df["RadialDistributionFunction|radial distribution function"][0]['distances'][:50]
    
    for i,d in enumerate(dist):
      df["RadialDistributionFunction|radial distribution function|d_{:.2f}".format(d)] = df["RadialDistributionFunction|radial distribution function"].apply(lambda x: x['distribution'][i])
    df = df.drop("RadialDistributionFunction|radial distribution function",axis=1)

    df["GlobalSymmetryFeatures|crystal_system"] = df["GlobalSymmetryFeatures|crystal_system"].map({"cubic":1, "tetragonal":2, "orthorombic":3, "hexagonal":4, "trigonal=":5, "monoclinic":6, "triclinic":7})
    df["GlobalSymmetryFeatures|is_centrosymmetric"] = df["GlobalSymmetryFeatures|is_centrosymmetric"].map({True:1, False:0})

    df = df.dropna(axis=1,how='all')
    df = df.replace([np.inf, -np.inf, np.nan], -1)
    df = df.select_dtypes(include='number')
    df_struct = df
    
    df = pd.DataFrame({'structure':struct,'composition':[x.composition for x in struct]})
    featurizer = MultipleFeaturizer([ElementProperty.from_preset("magpie"),
                                 OxidationStates(),
                                 AtomicOrbitals(),
                                 BandCenter(),
                                 ElectronegativityDiff(),
                                 ElectronAffinity(),
                                 Stoichiometry(),
                                 ValenceOrbital(),
                                 IonProperty(),
                                 ElementFraction(),
                                 TMetalFraction(),
                                 CohesiveEnergy(), # equals the formation energy
                                 Miedema(), # Formation enthalpies
                                 YangSolidSolution(),
                                 AtomicPackingEfficiency()
                                ])

    featurizer.featurize_dataframe(df,"composition",multiindex=True,ignore_errors=True)

    df = df.drop(['Input Data'],axis = 1)
    df = df.dropna(axis=1,how='all')
    df = df.replace([np.inf, -np.inf, np.nan], 0)
    df = df.select_dtypes(include='number')

    df.columns = df.columns.map('|'.join).str.strip('|')
    df_comp = df
    
    df = pd.DataFrame({'structure':struct,'composition':[x.composition for x in struct]})
    df_tot = df
    grdf = SiteStatsFingerprint(GeneralizedRadialDistributionFunction.from_preset('gaussian'),stats=('mean', 'std_dev')).fit(df_tot["structure"])
    df.columns = ["Input data|"+x for x in df.columns]
    
    SiteStatsFingerprint(AGNIFingerprints(),stats=('mean', 'std_dev')).featurize_dataframe(df,"Input data|structure",multiindex=False,ignore_errors=True)
    df.columns = ["AGNIFingerPrint|"+x if '|' not in x else x for x in df.columns]
    
    
    SiteStatsFingerprint(OPSiteFingerprint(),stats=('mean', 'std_dev')).featurize_dataframe(df,"Input data|structure",multiindex=False,ignore_errors=True)
    df.columns = ["OPSiteFingerprint|"+x if '|' not in x else x for x in df.columns]
    
    SiteStatsFingerprint(CrystalNNFingerprint.from_preset("ops"),stats=('mean', 'std_dev')).featurize_dataframe(df,"Input data|structure",multiindex=False,ignore_errors=True)
    df.columns = ["CrystalNNFingerprint|"+x if '|' not in x else x for x in df.columns]
    
    SiteStatsFingerprint(VoronoiFingerprint(),stats=('mean', 'std_dev')).featurize_dataframe(df,"Input data|structure",multiindex=False,ignore_errors=True)
    df.columns = ["VoronoiFingerprint|"+x if '|' not in x else x for x in df.columns]
    
    SiteStatsFingerprint(GaussianSymmFunc(),stats=('mean', 'std_dev')).featurize_dataframe(df,"Input data|structure",multiindex=False,ignore_errors=True)
    df.columns = ["GaussianSymmFunc" + x if '|' not in x else x for x in df.columns]
    
    SiteStatsFingerprint(ChemEnvSiteFingerprint.from_preset("simple"),stats=('mean', 'std_dev')).featurize_dataframe(df,"Input data|structure",multiindex=False,ignore_errors=True)
    df.columns = ["ChemEnvSiteFingerprint|"+x if '|' not in x else x for x in df.columns]
    
    SiteStatsFingerprint(CoordinationNumber(),stats=('mean', 'std_dev')).featurize_dataframe(df,"Input data|structure",multiindex=False,ignore_errors=True)
    df.columns = ["CoordinationNumber|"+x if '|' not in x else x for x in df.columns]
    
    grdf.featurize_dataframe(df,"Input data|structure",multiindex=False,ignore_errors=True)
    df.columns = ["GeneralizedRDF|"+x if '|' not in x else x for x in df.columns]
    
    #SiteStatsFingerprint(AngularFourierSeries.from_preset('gaussian'),stats=('mean', 'std_dev')).featurize_dataframe(df,"Input data|structure",multiindex=False,ignore_errors=True)
    #df.columns = ["AngularFourierSeries|"+x if '|' not in x else x for x in df.columns]
    
    SiteStatsFingerprint(LocalPropertyDifference(),stats=('mean', 'std_dev')).featurize_dataframe(df,"Input data|structure",multiindex=False,ignore_errors=True)
    df.columns = ["LocalPropertyDifference|"+x if '|' not in x else x for x in df.columns]
    
    SiteStatsFingerprint(BondOrientationalParameter(),stats=('mean', 'std_dev')).featurize_dataframe(df,"Input data|structure",multiindex=False,ignore_errors=True)
    df.columns = ["BondOrientationParameter|"+x if '|' not in x else x for x in df.columns]
    
    SiteStatsFingerprint(AverageBondLength(VoronoiNN()),stats=('mean', 'std_dev')).featurize_dataframe(df,"Input data|structure",multiindex=False,ignore_errors=True)
    df.columns = ["AverageBondLength|"+x if '|' not in x else x for x in df.columns]
    
    SiteStatsFingerprint(AverageBondAngle(VoronoiNN()),stats=('mean', 'std_dev')).featurize_dataframe(df,"Input data|structure",multiindex=False,ignore_errors=True)
    df.columns = ["AverageBondAngle|"+x if '|' not in x else x for x in df.columns]
    
    df = df.dropna(axis=1,how='all')
    df = df.loc[:, (df != 0).any(axis=0)]
    df = df.replace([np.inf, -np.inf, np.nan], -1)
    df = df.select_dtypes(include='number')
    df_site = df
    
    df = df_comp.join(df_struct.join(df_site))
    
    keyg = key_gen()
    keyg.add_all(step=20)
    df_X,df_Y = get_data(keys=keyg.get_keys())
    
    features_tot = []
    for i in range(df.shape[0]):
      features = []
      for x in features_li:
        if x in df.columns:
          val = (df[x][i]-df_X[x].min())/(df_X[x].max()-df_X[x].min())
          if val == float('inf') or val == float('-inf') or val == float('nan'):
            val = 0
          if val > 1:
            val = 1
          if val < 0:
            val = 0
          features.append(val)
        else:
          features.append(0)
      features_tot.append(np.array(features))
    return features_tot
  
def predict(features):
  y_min = np.load('min_y.npy')
  y_max = np.load('max_y.npy')
  y_pred = model.predict(np.array(features))
  y_pred = np.array(y_pred).flatten()
  y_pred = y_pred*(y_max-y_min)+y_min #scaling
  
  return y_pred

def predict_id(mp_id):
  mpr = MPRester(api_key="BsStVnxerc3bwToa")
  struct = mpr.get_structures(mp_id)[0]
  features = featurize_struct([struct])
  
  return predict(features)

def predict_polymorph(chem_form):
  mpr = MPRester(api_key="BsStVnxerc3bwToa")
  data = mpr.query(criteria=chem_form,properties=['formation_energy_per_atom','e_above_hull','structure','material_id'])
  ehull = [x['e_above_hull'] for x in data]
  idx = np.argsort(ehull)[:5]
  data_keep = []
  for i in idx:
    data_keep.append(data[i])
  struct = [x['structure'] for x in data_keep]
  e_form = [x['formation_energy_per_atom'] for x in data_keep]
  mp_id = [x['material_id'] for x in data_keep]
  
  pm = ['A','B','C','D','E']
  features = featurize_struct(struct)
  cst = (1.60218e-22 * 6.02214e23)
  res = []
  for f in features:
    res.append(predict([f])[41:81])

  fig,ax = plt.subplots()
  for i,r in enumerate(res):
    r = np.array(r)/cst
    r = smooth(r,5)
    r = savitzky_golay(r,5,3)
    ax.plot(range(5,800,20),(e_form[i]*1000)+r,label= "${}$ polymporph {}: {}".format(chem_form,pm[i],mp_id[i]),linewidth=3)
    ax.fill_between(range(5,800,20),(e_form[i]*1000)+r-mae_H,(e_form[i]*1000)+r+mae_H,alpha=0.2)
  #ax.set_ylim([-800,-600])
  ax.set_xlim([0,700])
  ax.legend()
  ax.set_ylabel('Helmolt free energy [meV/atom]')
  ax.set_xlabel('Temperature [K]')
  return res