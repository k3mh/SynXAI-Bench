import numpy as np
#import matplotlib.pylab as plab
#import matplotlib.pyplot as plt
#import seaborn as sns
import pandas as pd
from sklearn.preprocessing import minmax_scale


class dataset:
    def __init__(self):
        self.data = None
        self.meta_data = None
        self.features_names = None

    @staticmethod
    def __sigmoid__(self, x):
        return 1 / (1 + np.power(np.e, -x))

    @staticmethod
    def __generate_correlated_variables__(self, var, scale=.05):
        size = var.shape[0]
        z = np.random.normal(size=size, loc=1, scale=scale)
        y = var * z
        return y

    @staticmethod
    def logistic( x_vector):
        x_node = np.median(x_vector)
        L = 1
        K = 1
        # return [sigmoid(L/(1 + np.power(np.e, -K*(x - xNode)))) for x in xVector]
        return [L / (1 + np.power(np.e, -K * (x - x_node))) for x in x_vector]

    def get_gen_fn_list(self):
        funs = [self.generate_ds1 , self.generate_ds2, self.generate_ds3, self.generate_ds3_1, self.generate_ds4, self.generate_ds5]
        return funs

    # def generate_monotonic_dataset(self, size=5000):
    #     x = np.random.normal(loc=5, size=size)
    #
    #     # generate correlated variables
    #     scales = np.arange(.01, 0.21, .02)
    #     for scale in scales:
    #         y = generate_correlated_variables(var=x, scale=scale)
    #
    #         print(np.corrcoef(x, y)[0][1])
    #
    #     #  Generate noise variable
    #     scales = np.arange(1, 2, .1)
    #     for scale in scales:
    #         noise = generate_correlated_variables(var=x, scale=scale)
    #
    #         print(np.corrcoef(noise, y)[0][1], np.corrcoef(noise, x)[0][1])
    #
    #     plab.plot(y, (logistic(y)), 'bo')
    #     plab.show()


    def __generate_cor_vars__(self, samples=1000, corr=0.9):
        # Generate pearson correlated data with approximately cor(X, Y) = corr
        data = np.random.multivariate_normal([0, 0], [[1, corr], [corr, 1]], size=samples)
        X, Y = data[:, 0], data[:, 1]

        # That's it! Now let's take a look at the actual correlation:
        import scipy.stats as stats
        print('corr=', stats.pearsonr(X, Y)[0])
        return X, Y


    def generate_ds1(self, size=10000):
        #   Generate unrelated variables, any results will be false positive
        #   All variables are strictly positive and continuous

        ## randon variables x, y, z and v creation
        features = ["x1", "x2", "x3", "x4"]
        x1, x2 = self.__generate_cor_vars__(samples=size, corr=0.01)
        x3, x4 = self.__generate_cor_vars__(samples=size, corr=0.01)
        df = pd.DataFrame([x1, x2, x3, x4]).T
        df = pd.DataFrame(minmax_scale(df, axis=0, feature_range=(0, 1)), columns=features)
        df = df.astype(float)
        df = df.assign(y=df.x4.apply(lambda x: 1 if (x >= 0.5) else 0))
        df = df.drop("x4", axis=1)
        print(df.corr())
        self.meta_data = pd.DataFrame({"imp_vars": [[""]] * df.shape[0]})
        self.data = df
        features_names = df.columns.to_list()
        features_names.remove('y')
        self.features_names = features_names
        return self


    def generate_ds2(self, size=10000):
        #   Generate target correlated to one variable, target correlated to x4
        #   All variables are strictly positive and continuous

        ## randon variables x, y, z and v creation
        features = ["x1", "x2", "x3", "x4"]
        x1, x2 = self.__generate_cor_vars__(samples=size, corr=0.01)
        x3, x4 = self.__generate_cor_vars__(samples=size, corr=0.01)
        df = pd.DataFrame([x1, x2, x3, x4]).T
        df = pd.DataFrame(minmax_scale(df, axis=0, feature_range=(0, 1)), columns=features)
        df = df.astype(float)
        df = df.assign(y=df.x4.apply(lambda x: 1 if (x >= 0.5) else 0))
        print(df.corr())
        self.meta_data = pd.DataFrame({"imp_vars": [["x4"]] * df.shape[0]})
        self.data = df
        features_names = df.columns.to_list()
        features_names.remove('y')
        self.features_names = features_names
        return self


    def generate_ds3(self, size=10000):

        #  Generate target correlated to two variable, target is true when both x3, x4 are above 0.5. x3, and x4 are not correlated.
        #  All variables are strictly positive and continuous
        features = ["x1", "x2", "x3", "x4"]
        x1, x2 = self.__generate_cor_vars__(samples=size, corr=0.01)
        x3, x4 = self.__generate_cor_vars__(samples=size, corr=0.01)
        df = pd.DataFrame([x1, x2, x3, x4]).T
        df = pd.DataFrame(minmax_scale(df, axis=0, feature_range=(0, 1)), columns=features)
        df = df.astype(float)
        df = df.assign(y=df[["x3", "x4"]].apply(lambda x: 1 if (x[0] >= 0.5 and x[1] >= 0.5) else 0, axis=1))
        print(df.corr())
        self.meta_data = pd.DataFrame({"imp_vars": [["x4", "x3"]] * df.shape[0]})
        self.data = df
        features_names = df.columns.to_list()
        features_names.remove('y')
        self.features_names = features_names
        return self


    def generate_ds3_1(self, size=10000):
        #   Generate target correlated to two variable, target is true when both (x2^2 + 2*x3 + x4) >= 0.5.
        #   All variables are strictly positive and continuous

        features = ["x1", "x2", "x3", "x4"]
        x1, x2 = self.__generate_cor_vars__(samples=size, corr=0.01)
        x3, x4 = self.__generate_cor_vars__(samples=size, corr=0.01)
        df = pd.DataFrame([x1, x2, x3, x4]).T
        df = pd.DataFrame(minmax_scale(df, axis=0, feature_range=(0, 1)), columns=features)
        df = df.astype(float)
        tmp_var = df[["x2", "x3", "x4"]].apply(lambda x: np.power(x[0], 2) + (2 * x[1]) + x[2])
        df = df.assign(
            y=df[["x2", "x3", "x4"]].apply(lambda x: 1 if (np.power(x[0], 2) + (2 * x[1]) + x[2] >= 0.5) else 0, axis=1))
        print(df.corr())
        self.meta_data = pd.DataFrame({"imp_vars": [["x2", "x3", "x4"]] * df.shape[0]})
        self.data = df
        features_names = df.columns.to_list()
        features_names.remove('y')
        self.features_names = features_names
        return self


    def generate_ds4(self, size=10000):
        #   Generate target correlated to two variable, target is true when  x5 = 1  .
        #   All variables are strictly positive and continuous except x5 is categorical
        features = ["x1", "x2", "x3", "x4", "x5"]
        x1, x2 = self.__generate_cor_vars__(samples=size, corr=0.01)
        x3, x4 = self.__generate_cor_vars__(samples=size, corr=0.01)
        x5 = np.random.choice(2, size)
        df = pd.DataFrame([x1, x2, x3, x4, x5]).T
        df = pd.DataFrame(minmax_scale(df, axis=0, feature_range=(0, 1)), columns=features)
        df = df.astype(float)
        df = df.assign(y=df[["x5"]].apply(lambda x: 1 if (x[0] == 1) else 0, axis=1))
        print(df.corr())
        self.meta_data = pd.DataFrame({"imp_vars": [[ "x5"]] * df.shape[0]})
        self.data = df
        features_names = df.columns.to_list()
        features_names.remove('y')
        self.features_names = features_names
        return self


    def generate_ds5(self, size=10000):
        #   Generate target correlated to two variable, target is true when both x4, x5 are above 0.5 and = 1 respectively .
        #   All variables are strictly positive and continuous except x5 is categorical
        features = ["x1", "x2", "x3", "x4", "x5"]
        x1, x2 = self.__generate_cor_vars__(samples=size, corr=0.01)
        x3, x4 = self.__generate_cor_vars__(samples=size, corr=0.01)
        x5 = np.random.choice(2, size)
        df = pd.DataFrame([x1, x2, x3, x4, x5]).T
        df = pd.DataFrame(minmax_scale(df, axis=0, feature_range=(0, 1)), columns=features)
        df = df.astype(float)
        df = df.assign(y=df[["x4", "x5"]].apply(lambda x: 1 if (x[0] >= 0.5 and x[1] == 1) else 0, axis=1))
        print(df.corr())
        self.meta_data = pd.DataFrame({"imp_vars": [["x4", "x5"]] * df.shape[0]})
        self.data = df
        features_names = df.columns.to_list()
        features_names.remove('y')
        self.features_names = features_names
        return self

    def get_data(self):
        return self.data





