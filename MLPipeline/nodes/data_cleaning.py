from MLPipeline import Node
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functools import partial

class checksum(Node):
    def __init__(self, name="Check sum"):
        super().__init__(name)

        def nzcol(data, filename):
            X, y = data
            X = X.loc[:,X.astype(bool).sum(axis=0)>0]
            return (X,y)

        def sum100(data, filename):
            X, y = data
            mask = X.sum(axis=1).between(99,101)
            X = X[mask]
            y = y[mask]
            return (X,y)
        self.steps = [nzcol,sum100]


class shuffle_cols(Node):
    def __init__(self, name="Shuffle columns", x=None, y=None):
        super().__init__(name)
        self.x =x
        self.y =y
        def select_cols(data, filename):
            df = pd.concat(data, axis=1)
            names = df.columns.to_list()
            if self.x==None:
                self.x = [i for i in names if i not in self.y]
            if self.y==None:
                self.y = [i for i in names if i not in self.x]
            return (df[self.x], df[self.y])
        self.steps = [select_cols]

def make_hist(filename,y):
    fig,axs = plt.subplots(y.values.shape[1],1,figsize = (6, y.values.shape[1]*6), gridspec_kw={'hspace':0.2})
    for i in range(y.values.shape[1]):
        axs[i].hist(y.values[:,i],bins=10,label=y.columns[i])
        axs[i].set_xlabel(y.columns[i])
        axs[i].set_ylabel("Frequency")
    try:
        plt.savefig(filename+"FS_distribution.png")
    except:
        pass

class feature_selection(Node):
    def __init__(self, name="Selecting Components",num_glasses=30):
        super().__init__(name)

        def final_features(data, filename):
            X, y = data
            flag = True
            while flag:
                count_component  = X.astype(bool).sum(axis=0) <= num_glasses
                x30 = X.loc[:,~count_component]
                order = int(filename.split("/")[-1].split("_")[0])
                X, y = checksum()((x30, y), order)
                count_component  = X.astype(bool).sum(axis=0) <= num_glasses
                if count_component.sum()==0:
                    flag = False
            make_hist(filename,y)

            return X,y

        self.steps = [final_features]


class take_oxides(Node):
    def __init__(self, name="Taking only oxide components"):
        super().__init__(name)

        def get_oxides(data, filename, oxides=None):
            if type(oxides)==None:
                oxides = ['Ag2O','Al2O3','Am2O3','AmO2','As2O3','As2O5','Au2O','Au2O3','B2O3','BaO','BaO2','BeO','Bi2O3','Bi2O5','CaO','CdO','Ce2O3','CeO','CeO2','CO2','Co2O3','Co3O4','CoO','Cr2O3','Cr3O4','CrO','CrO3','Cs2O','Cu2O','CuO','Dy2O3','Er2O3','ErO2','Eu2O3','EuO','Fe2O3','Fe3O4','FeO','Ga2O3','Gd2O3','GeO','GeO2','H2O','HfO2','Hg2O','HgO','Ho2O3','In2O3','K2O','KAsO3','La2O3','Li2O','Lu2O3','MgO','Mn2O3','Mn2O7','Mn3O4','MnO','MnO2','Mo2O3','Mo2O5','MoO','MoO2','MoO3','N2O5','Na2O','Nb2O3','Nb2O5','Nd2O3','Ni2O3','NiO','NO2','NO3','Np2O3','NpO2','P2O3','P2O5','Pb3O4','PbO','PbO2','PdO','Pr2O3','Pr6O11','PrO2','PtO','PtO2','Pu2O3','PuO2','Rb2O','Re2O7','ReO3','Rh2O3','RhO2','RuO2','Sb2O3','Sb2O5','SbO','SbO2','Sc2O3','SeO2','SeO3','SiO','SiO2','Sm2O3','Sn2O3','SnO','SnO2','SO2','SO3','SO4','SrO','Ta2O3','Ta2O5','Tb2O3','Tb3O7','Tb4O7','TbO2','TcO2','TeO','TeO2','TeO3','ThO2','Ti2O3','TiO','TiO2','Tl2O','Tl2O3','Tm2O3','U2O5','U3O8','UO2','UO3','V2O3','V2O5','VO2','VO6','WO3','Y2O3','Yb2O3','YbO2','ZnO','ZrO2','CoO2','V2O4','D2O','OH','Pm2O3','U2O3']

            X, y = data
            oxide_names = list(set(X.columns.tolist()).intersection(set(oxides)))
            X = X[oxide_names]
            return (X,y)
        self.steps = [get_oxides]

class drop_duplicates(Node):
    import pandas as pd
    import numpy  as np
    def __init__(self, name="Duplicate removal",num_comp=30):
        super().__init__(name)

        def mean_dup(x_):
            def reject_outliers(data, m = 2.):
                d = np.abs(data - np.median(data))
                mdev = np.median(d)
                s = d/mdev if mdev else 0.
                return (s<m)

            if 1==len(np.unique(x_.values)):
                return x_.values[0]
            else:
                x = x_.values[reject_outliers(x_.values.copy())]
                x_mean = x.mean()
                mask = (x_mean*0.975 <= x) & (x <= x_mean*1.025)
                return x[mask].mean()

        def drop(data, filename):
            X, y = data
            X_columns, y_columns = X.columns, y.columns
            data = pd.concat((X,y),axis=1)
            data = data.groupby(list(X_columns),as_index=False).agg(mean_dup)
            data = data.dropna()
            X,y  = data[X_columns], data[y_columns]
            return (X,y)


        self.steps = [drop]


class data_spliting(Node):
    def __init__(self, name="train_test_split"):
        super().__init__(name)

        def manual_split(data, filename):
            X, y = data

            def train_test_split_manual(X,y,test_size = 0.2):
                from sklearn.model_selection import train_test_split
                X_tr, X_te, y_tr, y_te = train_test_split(X,y, test_size = 0.2)
                while (sum((X_tr[:] != 0).sum(0)>=(0.6*(X[:] != 0).sum(0))) != X_tr.shape[-1]):
                    X_tr, X_te, y_tr, y_te = train_test_split(X,y, test_size=0.2)
                return X_tr, X_te, y_tr, y_te

            x_train, x_test, y_train, y_test = train_test_split_manual(X, y, test_size=0.2)

            x_train.to_csv(filename+"_train_split_X.csv", index=False)
            y_train.to_csv(filename+"_train_split_y.csv", index=False)

            x_test.to_csv(filename+"_test_split_X.csv", index=False)
            y_test.to_csv(filename+"_test_split_y.csv", index=False)

            plt.hist(y_train.values,label="Train_%s"%y_train.columns[0], bins=10)
            plt.hist(y_test.values,label="Test_%s"%y_test.columns[0], bins=10)
            plt.grid(False)
            plt.legend()
            plt.title("")

            plt.savefig(filename+"tr_test_distribution.png")

            return (x_train,y_train)

        self.steps = [manual_split]

class normalize_data(Node):
    def __init__(self, name="Data Normalization", mean=None, std=None):
        super().__init__(name)
        self.mean = mean
        self.std = std
        def norm(data, filename, mean=None, std=None):
            X, y = data
            left_col, right_col = X.columns,y.columns
            data = pd.concat((X,y),axis = 1)
            if mean ==None and std==None:
                m = data.mean()
                s = data.std()
                mask = s<=0.0001
                s[mask] = 1
                import json
                ms = {'means':list(m.values),'stds':list(s.values)}
                with open(filename+'_means_and_stds.json','w') as f:
                    json.dump(ms,f)
            else:
                m = mean
                s = std
            data = (data-m)/s
            X_norm,y_norm = data[left_col],data[right_col]

            return (X_norm,y_norm)

        self.steps = [partial(norm,mean = self.mean,std = self.std)]
