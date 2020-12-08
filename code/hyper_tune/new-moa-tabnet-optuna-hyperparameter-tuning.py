#!/usr/bin/env python
# coding: utf-8

# In[40]:


# TabNet
get_ipython().system('pip install --no-index --find-links /kaggle/input/pytorchtabnet/pytorch_tabnet-2.0.0-py3-none-any.whl pytorch-tabnet')


# In[41]:


import sys
sys.path.append('input/iterativestratification')
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold


# In[42]:


### General ###
import os
import copy
import tqdm
import pickle
import random
import warnings
warnings.filterwarnings("ignore")
os.environ["CUDA_LAUNCH_BLOCKING"] = '1'

### Data Wrangling ###
import numpy as np
import pandas as pd
from scipy import stats

### Machine Learning ###
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

### Deep Learning ###
import torch
from torch import nn
import torch.optim as optim
from torch.nn import functional as F
from torch.nn.modules.loss import _WeightedLoss
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
# Tabnet 
from pytorch_tabnet.metrics import Metric
from pytorch_tabnet.tab_model import TabNetRegressor

from pickle import load,dump

### Make prettier the prints ###
from colorama import Fore
c_ = Fore.CYAN
m_ = Fore.MAGENTA
r_ = Fore.RED
b_ = Fore.BLUE
y_ = Fore.YELLOW
g_ = Fore.GREEN


# In[43]:


from sklearn.preprocessing import QuantileTransformer


# In[44]:


os.listdir('input/lish-moa')


# In[45]:


train_features = pd.read_csv('input/lish-moa/train_features.csv')
train_targets_scored = pd.read_csv('input/lish-moa/train_targets_scored.csv')
train_targets_nonscored = pd.read_csv('input/lish-moa/train_targets_nonscored.csv')

test_features = pd.read_csv('input/lish-moa/test_features.csv')
df = pd.read_csv('input/lish-moa/sample_submission.csv')


# In[46]:


train_features2=train_features.copy()
test_features2=test_features.copy()


# In[47]:


GENES = [col for col in train_features.columns if col.startswith('g-')]
CELLS = [col for col in train_features.columns if col.startswith('c-')]


# In[48]:


qt = QuantileTransformer(n_quantiles=100,random_state=42,output_distribution='normal')
train_features[GENES+CELLS] = qt.fit_transform(train_features[GENES+CELLS])
test_features[GENES+CELLS] = qt.transform(test_features[GENES+CELLS])


# In[49]:


seed = 42

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
set_seed(seed)


# In[50]:


n_comp = 600  #<--Update
pca_g = PCA(n_components=n_comp, random_state=42)
data = pd.concat([pd.DataFrame(train_features[GENES]), pd.DataFrame(test_features[GENES])])
gpca= (pca_g.fit(data[GENES]))
train2= (gpca.transform(train_features[GENES]))
test2 = (gpca.transform(test_features[GENES]))

train_gpca = pd.DataFrame(train2, columns=[f'pca_G-{i}' for i in range(n_comp)])
test_gpca = pd.DataFrame(test2, columns=[f'pca_G-{i}' for i in range(n_comp)])

# drop_cols = [f'c-{i}' for i in range(n_comp,len(GENES))]
train_features = pd.concat((train_features, train_gpca), axis=1)
test_features = pd.concat((test_features, test_gpca), axis=1)

dump(gpca, open('gpca.pkl', 'wb'))


# In[51]:


train_gpca.head()


# In[52]:


gpca000= (pca_g.fit(train_features[GENES]))
train000= (gpca000.transform(train_features[GENES]))
train_gpca000 = pd.DataFrame(train000, columns=[f'pca_G-{i}' for i in range(n_comp)])
train_gpca000.shape


# In[53]:


train_gpca000.head()


# In[54]:


train_gpca.head()


# In[55]:


#CELLS
n_comp = 50  #<--Update

pca_c = PCA(n_components=n_comp, random_state=42)
data = pd.concat([pd.DataFrame(train_features[CELLS]), pd.DataFrame(test_features[CELLS])])
cpca= (pca_c.fit(data[CELLS]))
train2= (cpca.transform(train_features[CELLS]))
test2 = (cpca.transform(test_features[CELLS]))

train_cpca = pd.DataFrame(train2, columns=[f'pca_C-{i}' for i in range(n_comp)])
test_cpca = pd.DataFrame(test2, columns=[f'pca_C-{i}' for i in range(n_comp)])

# drop_cols = [f'c-{i}' for i in range(n_comp,len(CELLS))]
train_features = pd.concat((train_features, train_cpca), axis=1)
test_features = pd.concat((test_features, test_cpca), axis=1)

dump(cpca, open('cpca.pkl', 'wb'))


# In[56]:


from sklearn.feature_selection import VarianceThreshold

c_n = [f for f in list(train_features.columns) if f not in ['sig_id', 'cp_type', 'cp_time', 'cp_dose']]
mask = (train_features[c_n].var() >= 0.85).values
tmp = train_features[c_n].loc[:, mask]
train_features = pd.concat([train_features[['sig_id', 'cp_type', 'cp_time', 'cp_dose']], tmp], axis=1)
tmp = test_features[c_n].loc[:, mask]
test_features = pd.concat([test_features[['sig_id', 'cp_type', 'cp_time', 'cp_dose']], tmp], axis=1)


# In[57]:


from sklearn.cluster import KMeans
def fe_cluster_genes(train, test, n_clusters_g = 22, SEED = 42):
    
    features_g = GENES
    #features_c = CELLS
    
    def create_cluster(train, test, features, kind = 'g', n_clusters = n_clusters_g):
        train_ = train[features].copy()
        test_ = test[features].copy()
        data = pd.concat([train_, test_], axis = 0)
        kmeans_genes = KMeans(n_clusters = n_clusters, random_state = SEED).fit(data)
        dump(kmeans_genes, open('kmeans_genes.pkl', 'wb'))
        train[f'clusters_{kind}'] = kmeans_genes.predict(train_.values)
        test[f'clusters_{kind}'] = kmeans_genes.predict(test_.values)
        train = pd.get_dummies(train, columns = [f'clusters_{kind}'])
        test = pd.get_dummies(test, columns = [f'clusters_{kind}'])
        return train, test
    
    train, test = create_cluster(train, test, features_g, kind = 'g', n_clusters = n_clusters_g)
   # train, test = create_cluster(train, test, features_c, kind = 'c', n_clusters = n_clusters_c)
    return train, test

train_features2 ,test_features2=fe_cluster_genes(train_features2,test_features2)


# In[58]:


def fe_cluster_cells(train, test, n_clusters_c = 4, SEED = 42):
    
    #features_g = GENES
    features_c = CELLS
    
    def create_cluster(train, test, features, kind = 'c', n_clusters = n_clusters_c):
        train_ = train[features].copy()
        test_ = test[features].copy()
        data = pd.concat([train_, test_], axis = 0)
        kmeans_cells = KMeans(n_clusters = n_clusters, random_state = SEED).fit(data)
        dump(kmeans_cells, open('kmeans_cells.pkl', 'wb'))
        train[f'clusters_{kind}'] = kmeans_cells.predict(train_.values)
        test[f'clusters_{kind}'] = kmeans_cells.predict(test_.values)
        train = pd.get_dummies(train, columns = [f'clusters_{kind}'])
        test = pd.get_dummies(test, columns = [f'clusters_{kind}'])
        return train, test
    
   # train, test = create_cluster(train, test, features_g, kind = 'g', n_clusters = n_clusters_g)
    train, test = create_cluster(train, test, features_c, kind = 'c', n_clusters = n_clusters_c)
    return train, test

train_features2 ,test_features2=fe_cluster_cells(train_features2,test_features2)


# In[59]:


train_pca=pd.concat((train_gpca,train_cpca),axis=1)
test_pca=pd.concat((test_gpca,test_cpca),axis=1)


# In[60]:


def fe_cluster_pca(train, test,n_clusters=5,SEED = 42):
        data=pd.concat([train,test],axis=0)
        kmeans_pca = KMeans(n_clusters = n_clusters, random_state = SEED).fit(data)
        dump(kmeans_pca, open('kmeans_pca.pkl', 'wb'))
        train[f'clusters_pca'] = kmeans_pca.predict(train.values)
        test[f'clusters_pca'] = kmeans_pca.predict(test.values)
        train = pd.get_dummies(train, columns = [f'clusters_pca'])
        test = pd.get_dummies(test, columns = [f'clusters_pca'])
        return train, test
train_cluster_pca ,test_cluster_pca = fe_cluster_pca(train_pca,test_pca)


# In[61]:


train_cluster_pca = train_cluster_pca.iloc[:,650:]
test_cluster_pca = test_cluster_pca.iloc[:,650:]


# In[62]:


train_features_cluster=train_features2.iloc[:,876:]
test_features_cluster=test_features2.iloc[:,876:]


# In[63]:



gsquarecols=['g-574','g-211','g-216','g-0','g-255','g-577','g-153','g-389','g-60','g-370','g-248','g-167','g-203','g-177','g-301','g-332','g-517','g-6','g-744','g-224','g-162','g-3','g-736','g-486','g-283','g-22','g-359','g-361','g-440','g-335','g-106','g-307','g-745','g-146','g-416','g-298','g-666','g-91','g-17','g-549','g-145','g-157','g-768','g-568','g-396']


# In[64]:


def fe_stats(train, test):
    
    features_g = GENES
    features_c = CELLS
    
    for df in train, test:
        df['g_sum'] = df[features_g].sum(axis = 1)
        df['g_mean'] = df[features_g].mean(axis = 1)
        df['g_std'] = df[features_g].std(axis = 1)
        df['g_kurt'] = df[features_g].kurtosis(axis = 1)
        df['g_skew'] = df[features_g].skew(axis = 1)
        df['c_sum'] = df[features_c].sum(axis = 1)
        df['c_mean'] = df[features_c].mean(axis = 1)
        df['c_std'] = df[features_c].std(axis = 1)
        df['c_kurt'] = df[features_c].kurtosis(axis = 1)
        df['c_skew'] = df[features_c].skew(axis = 1)
        df['gc_sum'] = df[features_g + features_c].sum(axis = 1)
        df['gc_mean'] = df[features_g + features_c].mean(axis = 1)
        df['gc_std'] = df[features_g + features_c].std(axis = 1)
        df['gc_kurt'] = df[features_g + features_c].kurtosis(axis = 1)
        df['gc_skew'] = df[features_g + features_c].skew(axis = 1)
        
        df['c52_c42'] = df['c-52'] * df['c-42']
        df['c13_c73'] = df['c-13'] * df['c-73']
        df['c26_c13'] = df['c-23'] * df['c-13']
        df['c33_c6'] = df['c-33'] * df['c-6']
        df['c11_c55'] = df['c-11'] * df['c-55']
        df['c38_c63'] = df['c-38'] * df['c-63']
        df['c38_c94'] = df['c-38'] * df['c-94']
        df['c13_c94'] = df['c-13'] * df['c-94']
        df['c4_c52'] = df['c-4'] * df['c-52']
        df['c4_c42'] = df['c-4'] * df['c-42']
        df['c13_c38'] = df['c-13'] * df['c-38']
        df['c55_c2'] = df['c-55'] * df['c-2']
        df['c55_c4'] = df['c-55'] * df['c-4']
        df['c4_c13'] = df['c-4'] * df['c-13']
        df['c82_c42'] = df['c-82'] * df['c-42']
        df['c66_c42'] = df['c-66'] * df['c-42']
        df['c6_c38'] = df['c-6'] * df['c-38']
        df['c2_c13'] = df['c-2'] * df['c-13']
        df['c62_c42'] = df['c-62'] * df['c-42']
        df['c90_c55'] = df['c-90'] * df['c-55']
        
        
        for feature in features_c:
             df[f'{feature}_squared'] = df[feature] ** 2     
                
        for feature in gsquarecols:
            df[f'{feature}_squared'] = df[feature] ** 2        
        
    return train, test

train_features2,test_features2=fe_stats(train_features2,test_features2)


# In[65]:


train_features_stats=train_features2.iloc[:,902:]
test_features_stats=test_features2.iloc[:,902:]


# In[66]:


train_features = pd.concat((train_features, train_features_cluster,train_cluster_pca,train_features_stats), axis=1)
test_features = pd.concat((test_features, test_features_cluster,test_cluster_pca,test_features_stats), axis=1)


# In[67]:


train = train_features.merge(train_targets_scored, on='sig_id')
train = train[train['cp_type']!='ctl_vehicle'].reset_index(drop=True)
test = test_features[test_features['cp_type']!='ctl_vehicle'].reset_index(drop=True)

target = train[train_targets_scored.columns]


# In[68]:


train = train.drop('cp_type', axis=1)
test = test.drop('cp_type', axis=1)


# In[69]:


target_cols = target.drop('sig_id', axis=1).columns.values.tolist()


# In[70]:


target=target[target_cols]


# In[71]:


train = pd.get_dummies(train, columns=['cp_time','cp_dose'])
test_ = pd.get_dummies(test, columns=['cp_time','cp_dose'])


# In[72]:


feature_cols = [c for c in train.columns if c not in target_cols]
feature_cols = [c for c in feature_cols if c not in ['sig_id']]


# In[73]:


train = train[feature_cols]
test = test_[feature_cols]


# In[74]:


X_test = test.values


# In[75]:


from torch.nn.modules.loss import _WeightedLoss
class SmoothBCEwLogits(_WeightedLoss):
    def __init__(self, weight=None, reduction='mean', smoothing=0.0):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    @staticmethod
    def _smooth(targets:torch.Tensor, n_labels:int, smoothing=0.0):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            targets = targets * (1.0 - smoothing) + 0.5 * smoothing
        return targets

    def forward(self, inputs, targets):
        targets = SmoothBCEwLogits._smooth(targets, inputs.size(-1),
            self.smoothing)
        loss = F.binary_cross_entropy_with_logits(inputs, targets,self.weight)

        if  self.reduction == 'sum':
            loss = loss.sum()
        elif  self.reduction == 'mean':
            loss = loss.mean()

        return loss


# In[76]:


class LogitsLogLoss(Metric):

    def __init__(self):
        self._name = "logits_ll"
        self._maximize = False

    def __call__(self, y_true, y_pred):
        logits = 1 / (1 + np.exp(-y_pred))
        aux = (1 - y_true) * np.log(1 - logits + 5e-5) + y_true * np.log(logits + 5e-5)
        return np.mean(-aux)


# In[77]:


'''MAX_EPOCH = 200

tabnet_params = dict(
    n_d = 32,
    n_a = 32,
    n_steps = 1,
    gamma = 1.3,
    lambda_sparse = 0,
    optimizer_fn = optim.Adam,
    optimizer_params = dict(lr = 2e-2, weight_decay = 1e-5),
    mask_type = "entmax",
    scheduler_params = dict(mode = "min", patience = 5, min_lr = 1e-5, factor = 0.9),
    scheduler_fn = ReduceLROnPlateau,
    seed = seed,
    verbose = 10
)'''


# In[88]:


'''scores_auc_all = []
test_cv_preds = []

NB_SPLITS = 10
mskf = MultilabelStratifiedKFold(n_splits = NB_SPLITS, random_state = 0, shuffle = True)

oof_preds = []
oof_targets = []
scores = []
scores_auc = []
SEED = [20,21,22]
for s in SEED:
    tabnet_params['seed'] = s
    for fold_nb, (train_idx, val_idx) in enumerate(mskf.split(train, target)):
        print(b_,"FOLDS: ", r_, fold_nb + 1, y_, 'seed:', tabnet_params['seed'])
        print(g_, '*' * 60, c_)
    
        X_train, y_train = train.values[train_idx, :], target.values[train_idx, :]
        X_val, y_val = train.values[val_idx, :], target.values[val_idx, :]
        ### Model ###
        model = TabNetRegressor(**tabnet_params)
        
        ### Fit ###
        model.fit(
            X_train = X_train,
            y_train = y_train,
            eval_set = [(X_val, y_val)],
            eval_name = ["val"],
            eval_metric = ["logits_ll"],
            max_epochs = MAX_EPOCH,
            patience = 20,
            batch_size = 1024, 
            virtual_batch_size = 32,
            num_workers = 1,
            drop_last = False,
            loss_fn = SmoothBCEwLogits(smoothing=5e-5))
        print(y_, '-' * 60)
    
        ### Predict on validation ###
        preds_val = model.predict(X_val)
        # Apply sigmoid to the predictions
        preds = 1 / (1 + np.exp(-preds_val))
        score = np.min(model.history["val_logits_ll"])
        saving_path_name = 'TabNet_seed_'+str(tabnet_params['seed'])+'_fold_'+str(fold_nb+1)
        saved_filepath = model.save_model(saving_path_name)
        
        loaded_model =  TabNetRegressor()
        loaded_model.load_model(saved_filepath)
    
        ### Save OOF for CV ###
        oof_preds.append(preds_val)
        oof_targets.append(y_val)
        scores.append(score)
    
        ### Predict on test ###
        model.load_model(saved_filepath)
        preds_test = model.predict(X_test)
        test_cv_preds.append(1 / (1 + np.exp(-preds_test)))

oof_preds_all = np.concatenate(oof_preds)
oof_targets_all = np.concatenate(oof_targets)
test_preds_all = np.stack(test_cv_preds)'''


# In[ ]:





# In[ ]:

print(m_,'batch_size:',512, m_,'virtual_size',64)
#print('virtual_size:',64)

# Low number of epochs for Optuna
MAX_EPOCH=200
def objective(trial):
    SEED = 25  # [20,21,22]
     # all hyperparameters here
#     mask_type = trial.suggest_categorical("mask_type", ["entmax", "sparsemax"])
    n_d = trial.suggest_int("n_d", 8, 32, step=8)
    n_a = trial.suggest_int("n_a", 32, 64, step=8)
    #n_steps = trial.suggest_int("n_steps", 1, 3, step=1)

    #n_shared = trial.suggest_int("n_shared", 1, 2)
    #n_independent = trial.suggest_int("n_independent", 1, 2, step=1)
    clip_value = trial.suggest_int("clip_value", 1, 2, step=1)

    gamma = trial.suggest_float("gamma", 1., 1.6, step=0.2)
    #lambda_sparse = trial.suggest_float("lambda_sparse", 1e-6, 1e-3, log=True)
    #batch_size = trial.suggest_int("batch_size", 512, 1024, step=256)
    #momentum = trial.suggest_float("momentum", 0.02, 0.1, step=0.02)

    #factor = trial.suggest_float("factor", 0.5, 0.9,step=0.1)

    for t in trial.study.trials:
        if t.state != optuna.structs.TrialState.COMPLETE:
            continue

        if t.params == trial.params:
            return t.value  # Return the previous value without re-evaluating it.
    
    tabnet_params = dict(n_d=n_d, 
                         n_a=n_a, 
                         n_steps=1,
                         gamma=gamma,
                         n_shared=1,
                         n_independent=1,
                         lambda_sparse=0,
                         #momentum = momentum,
                         clip_value=clip_value,
                         optimizer_fn=torch.optim.Adam,
                         optimizer_params=dict(lr=2e-2, weight_decay=1e-5),
                         mask_type="entmax", 
                         scheduler_params=dict(mode="min",
                                               patience=5,
                                               min_lr=1e-5,
                                               factor=0.5,),
                         scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
                         verbose=10,
                         seed=SEED,
                         )
    print(m_,'params:',tabnet_params)
    scores_auc_all = []
    test_cv_preds = []

    NB_SPLITS = 7 # 7
    mskf = MultilabelStratifiedKFold(n_splits = NB_SPLITS, random_state = 0, shuffle = True)

    oof_preds = []
    oof_targets = []
    scores = []
    scores_auc = []
    for fold_nb, (train_idx, val_idx) in enumerate(mskf.split(train, target)):
        print(b_,"FOLDS: ", r_, fold_nb + 1)
        print(b_, "FOLDS: ", r_, fold_nb + 1, y_, 'seed:', tabnet_params['seed'])
        print(g_, '*' * 60, c_)

        X_train, y_train = train.values[train_idx, :], target.values[train_idx, :]
        X_val, y_val = train.values[val_idx, :], target.values[val_idx, :]
        ### Model ###
        model =  TabNetRegressor(**tabnet_params)
       
        ### Fit ###
        model.fit(
            X_train = X_train,
            y_train = y_train,
            eval_set = [(X_val, y_val)],
            eval_name = ["val"],
            eval_metric = ["logits_ll"],
            max_epochs = MAX_EPOCH,
            patience = 20,
            batch_size = 512, #1024
            virtual_batch_size = 64,#32
            num_workers = 1,
            drop_last = False,
            # To use binary cross entropy because this is not a regression problem
            loss_fn = SmoothBCEwLogits(smoothing =0.0005)
        )
        print(y_, '-' * 60)

        ### Predict on validation ###
        preds_val = model.predict(X_val)
        # Apply sigmoid to the predictions
        preds = 1 / (1 + np.exp(-preds_val))
        score = np.min(model.history["val_logits_ll"])

        ### Save OOF for CV ###
        oof_preds.append(preds_val)
        oof_targets.append(y_val)
        scores.append(score)

        ### Predict on test ###
        #preds_test = model.predict(X_test)
        #test_cv_preds.append(1 / (1 + np.exp(-preds_test)))

    #oof_preds_all = np.concatenate(oof_preds)
    #oof_targets_all = np.concatenate(oof_targets)
    #test_preds_all = np.stack(test_cv_preds)
    return np.mean(scores)


# In[ ]:


import optuna
pruner = optuna.pruners.MedianPruner() 

study = optuna.create_study(direction="minimize", pruner=pruner)
#study.optimize(objective, n_jobs=1, n_trials=25, gc_after_trial=True)

unique_trials = 50
while unique_trials > len(set(str(t.params) for t in study.trials)):
    study.optimize(objective, n_jobs=1, n_trials=50, gc_after_trial=True)


print("Number of finished trials: {}".format(len(study.trials)))

print("Best trial:")
trial = study.best_trial

print("  Value: {}".format(trial.value))

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))


# In[ ]:


hist = study.trials_dataframe()
hist.to_csv('hist_512.csv', index=False)


# In[ ]:


print('Best trial: score {},\nparams {}'.format(study.best_trial.value,study.best_trial.params))


# In[ ]:
import plotly
fig_hist = optuna.visualization.plot_optimization_history(study)
plotly.offline.plot(fig_hist, filename='./plot_history.html', auto_open=False)


# In[ ]:


fig_slice = optuna.visualization.plot_slice(study)
plotly.offline.plot(fig_slice, filename='./plot_slice.html', auto_open=False)


# In[ ]:


fig_contour = optuna.visualization.plot_contour(study)
plotly.offline.plot(fig_contour, filename='./plot_contour.html', auto_open=False)

# In[ ]:


fig_coor=optuna.visualization.plot_parallel_coordinate(study)
plotly.offline.plot(fig_coor, filename='./plot_coor.html', auto_open=False)
# In[ ]:


best_params=study.best_trial.params
best_params

