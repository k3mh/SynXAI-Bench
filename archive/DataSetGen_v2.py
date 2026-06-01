import numpy as np
#import matplotlib.pylab as plab
#import matplotlib.pyplot as plt
#import seaborn as sns
import pandas as pd
from sklearn.preprocessing import minmax_scale
from sklearn.datasets import make_gaussian_quantiles, make_hastie_10_2, make_classification, make_friedman1,\
    make_friedman2, make_friedman3, make_blobs, make_moons
import logging

logger = logging.getLogger()

class dataset:
    num_all_features = 48
    features = ['x'+ str(i) for i in range(1, num_all_features + 1)]

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
        funs = [self.generate_ds0, self.generate_ds1, self.generate_ds2, self.generate_ds3, self.generate_ds4, self.generate_ds5,
                self.generate_ds6, self.generate_ds7, self.generate_ds8, self.generate_ds9, self.generate_ds10,
                self.generate_ds11, self.generate_ds12]

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
        logger.info("corr= {stats.pearsonr(X, Y)[0]}")
        return X, Y

    def __generate_cor_vars__lst(self, samples=1000, corr=0.9, num_vars=4):
        # Generate pearson correlated data with approximately cor(X, Y) = corr
        mean_lst = [0] *  num_vars
        cov_mtx = np.empty([num_vars, num_vars])
        cov_mtx.fill(corr)
        np.fill_diagonal(cov_mtx, 1)

        data = np.random.multivariate_normal(mean_lst, cov_mtx, size=samples)

        # That's it! Now let's take a look at the actual correlation:
        corr_ = np.corrcoef(data.T)
        np.fill_diagonal(corr_, 0)
        logger.info(f"shape:  {corr_.shape}")
        logger.info(f"max_ correlation  {corr_.max()}")
        return data

    def generate_ds0(self, size=10000):
        #   Generate unrelated variables, any results will be false positive
        #   All variables are strictly positive and continuous

        ## randon variables x, y, z and v creation
        features = ['x'+ str(i) for i in range(1, self.num_all_features + 2)]
        data =  self.__generate_cor_vars__lst(samples=size, corr=0, num_vars=len(features))
        df = pd.DataFrame(data, columns=features)

        #df = pd.DataFrame([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12]).T
        # corr_mtx = df.corr().values
        # np.fill_diagonal(corr_mtx, 0)
        # print("maximum correlation:", corr_mtx.max())
        y_temp_col = 'x' + str(self.num_all_features + 1)

        df = pd.DataFrame(minmax_scale(df, axis=0, feature_range=(0, 1)), columns=features)
        df = df.astype(float)
        df = df.assign(y=df[y_temp_col].apply(lambda x: 1 if (x >= 0.5) else 0))
        df = df.drop(y_temp_col, axis=1)

        self.meta_data = pd.DataFrame({"imp_vars": [[""]] * df.shape[0] , "RGS": ["RGS0"] * df.shape[0]})
        self.data = df
        features_names = df.columns.to_list()
        features_names.remove('y')
        self.features_names = features_names
        return self


    def generate_ds1(self, size=10000):
        #   Generate target correlated to one variable, target correlated to x11
        #   All variables are stri5ctly positive and continuous

        ## randon variables x, y, z and v creation
        # features = ["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12"]

        data = self.__generate_cor_vars__lst(samples=size, corr=0, num_vars=len(dataset.features))
        df = pd.DataFrame(data, columns=dataset.features)

        df = pd.DataFrame(minmax_scale(df, axis=0, feature_range=(0, 1)), columns=dataset.features)
        df = df.astype(float)
        df.x1 = df.x1 * 20
        median_point = df.x1.median()
        df = df.assign(y=df.x1.apply(lambda x: 1 if (x >= median_point) else 0))
        # print(df.corr())
        self.meta_data = pd.DataFrame({"imp_vars": [["x1"]] * df.shape[0] , "RGS": ["RGS1"] * df.shape[0]})
        self.data = df
        features_names = df.columns.to_list()
        features_names.remove('y')
        self.features_names = features_names
        return self


    def generate_ds2(self, size=10000):

        #  Generate target correlated to two variable, target is true when both x3, x4 are above 0.5. x3, and x4 are not correlated.
        #  All variables are strictly positive and continuous
        # features = ["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12"]

        data = self.__generate_cor_vars__lst(samples=size, corr=0, num_vars=len(dataset.features))
        df = pd.DataFrame(data, columns=dataset.features)

        df = pd.DataFrame(minmax_scale(df, axis=0, feature_range=(0, 1)), columns=dataset.features)
        df = df.astype(float)
        df[["x2", "x3"]] = df[["x2", "x3"]] * 20
        mid_point = (df.x2 + df.x3).median()
        df = df.assign(y=df[["x2", "x3"]].apply(lambda x: 1 if ((x[0] + x[1]) >= mid_point) else 0, axis=1))
        # print(df.corr())
        self.meta_data = pd.DataFrame({"imp_vars": [["x2", "x3"]] * df.shape[0] , "RGS": ["RGS2"] * df.shape[0]})
        self.data = df
        features_names = df.columns.to_list()
        features_names.remove('y')
        self.features_names = features_names
        return self

    def generate_ds3(self, size=10000):
        #   Generate target correlated to two variable, target is true when both (x2^2 + 2*x3 + x4) >= 0.5.
        #   All variables are strictly positive and continuous

        # features = ["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12"]

        data = self.__generate_cor_vars__lst(samples=size, corr=0, num_vars=len(dataset.features))
        df = pd.DataFrame(data, columns=dataset.features)

        df = pd.DataFrame(minmax_scale(df, axis=0, feature_range=(0, 1)), columns=dataset.features)
        df = df.astype(float)
        df[["x4", "x5", "x6"]] = df[["x4", "x5", "x6"]] * 20
        mid_point = df[["x4", "x5", "x6"]].apply(lambda x: (np.power(x[0], 2) + (2 * x[1]) + x[2])).median()
        df = df.assign(
            y=df[["x4", "x5", "x6"]].apply(lambda x: 1 if (((np.power(x[0], 2) + (2 * x[1]) + x[2])) >= mid_point) else 0, axis=1))
        # print(df.corr())
        self.meta_data = pd.DataFrame({"imp_vars": [["x4", "x5", "x6"]] * df.shape[0], "RGS": ["RGS3"] * df.shape[0]})
        self.data = df
        features_names = df.columns.to_list()
        features_names.remove('y')
        self.features_names = features_names
        return self

    def generate_ds4(self, size=10000):
        """
        Based on sklearn.datasets.make_gaussian_quantiles function. Based on two features.
        dataset is constructed by taking a multi-dimensional standard normal distribution and \
        defining classes separated by nested concentric multi-dimensional spheres such that \
         roughly equal numbers of samples are in each class .
        """


        data = self.__generate_cor_vars__lst(samples=size, corr=0, num_vars=len(dataset.features))
        df = pd.DataFrame(data, columns=dataset.features)

        df = pd.DataFrame(minmax_scale(df, axis=0, feature_range=(0, 1)), columns=dataset.features)

        temp_arr = make_gaussian_quantiles(n_features=2, n_classes=2, n_samples=size)
        temp_pd = pd.DataFrame({"x1": temp_arr[0].transpose()[0], "x2": temp_arr[0].transpose()[1], "y": temp_arr[1]})

        df.x7 = temp_pd.x1
        df.x8 = temp_pd.x2
        df    = df.assign(y=temp_pd.y)
        
        self.meta_data = pd.DataFrame({"imp_vars": [[ "x7", "x8"]] * df.shape[0] , "RGS": ["RGS4"] * df.shape[0]})
        self.data = df
        features_names = df.columns.to_list()
        features_names.remove('y')
        self.features_names = features_names
        return self

    def generate_ds5(self, size=10000):
        """
        Based on sklearn.datasets.make_gaussian_quantiles function. Based on Five features.
        dataset is constructed by taking a multi-dimensional standard normal distribution and \
        defining classes separated by nested concentric multi-dimensional spheres such that \
         roughly equal numbers of samples are in each class .
        """

        data = self.__generate_cor_vars__lst(samples=size, corr=0, num_vars=len(dataset.features))
        df = pd.DataFrame(data, columns=dataset.features)

        df = pd.DataFrame(minmax_scale(df, axis=0, feature_range=(0, 1)), columns=dataset.features)

        temp_arr = make_gaussian_quantiles(n_features=5, n_classes=2, n_samples=size)
        temp_pd = pd.DataFrame({"x1": temp_arr[0].transpose()[0],\
                                "x2": temp_arr[0].transpose()[1], \
                                "x3": temp_arr[0].transpose()[2], \
                                "x4": temp_arr[0].transpose()[3], \
                                "x5": temp_arr[0].transpose()[4], \
                                "y": temp_arr[1]})

        df.x9  = temp_pd.x1
        df.x10 = temp_pd.x2
        df.x11 = temp_pd.x3
        df.x12 = temp_pd.x4
        df.x13 = temp_pd.x5

        df = df.assign(y=temp_pd.y)
        df.y.where(df.y == 1, 0, inplace=True)


        self.meta_data = pd.DataFrame({"imp_vars": [["x9", "x10", "x11", "x12", "x13"]] * df.shape[0], "RGS": ["RGS5"] * df.shape[0]})
        self.data = df
        features_names = df.columns.to_list()
        features_names.remove('y')
        self.features_names = features_names
        return self

    def generate_ds6(self, size=10000):
        """
        Based on sklearn.datasets.make_hastie_10_2 function. Based on 10 features.
        Generates data for binary classification used in Hastie et al. 2009, Example 10.2.
        The ten features are standard independent Gaussian and the target y is defined by:
        y[i] = 1 if np.sum(X[i] ** 2) > 9.34 else -1
        """

        data = self.__generate_cor_vars__lst(samples=size, corr=0, num_vars=len(dataset.features))
        df = pd.DataFrame(data, columns=dataset.features)

        df = pd.DataFrame(minmax_scale(df, axis=0, feature_range=(0, 1)), columns=dataset.features)

        temp_arr = make_hastie_10_2(n_samples=size)
        temp_pd = pd.DataFrame({"x1": temp_arr[0].transpose()[0], \
                                "x2": temp_arr[0].transpose()[1], \
                                "x3": temp_arr[0].transpose()[2], \
                                "x4": temp_arr[0].transpose()[3], \
                                "x5": temp_arr[0].transpose()[4], \
                                "x6": temp_arr[0].transpose()[5], \
                                "x7": temp_arr[0].transpose()[6], \
                                "x8": temp_arr[0].transpose()[7], \
                                "x9": temp_arr[0].transpose()[8], \
                                "x10": temp_arr[0].transpose()[9], \
                                "y": temp_arr[1]})

        df.x14 = temp_pd.x1
        df.x15 = temp_pd.x2
        df.x16 = temp_pd.x3
        df.x17 = temp_pd.x4
        df.x18 = temp_pd.x5
        df.x19 = temp_pd.x6
        df.x20 = temp_pd.x7
        df.x21 = temp_pd.x8
        df.x22 = temp_pd.x9
        df.x23 = temp_pd.x10
        df = df.assign(y=temp_pd.y)
        df.y.where(df.y == 1, 0, inplace=True)

        self.meta_data = pd.DataFrame(
            {"imp_vars": [["x14", "x15", "x16", "x17", "x18", "x19", "x20", "x21", "x22", "x23"]] * df.shape[0], "RGS": ["RGS5"] * df.shape[0]})
        self.data = df
        features_names = df.columns.to_list()
        features_names.remove('y')
        self.features_names = features_names
        return self



    def generate_ds7(self, size=10000):
        #   Generate target correlated to two variable, target is true when both 5*x3 or (x4 and x5) are above 0.5 and = 1 respectively .
        #   All variables are strictly positive and continuous except x5 is categorical
        # features = ["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12"]

        data = self.__generate_cor_vars__lst(samples=size, corr=0, num_vars=len(dataset.features))
        df = pd.DataFrame(data, columns=dataset.features)


        df = pd.DataFrame(minmax_scale(df, axis=0, feature_range=(0, 1)), columns=dataset.features)

        df = df.astype(float)
        temp_arr = make_friedman1(n_features=5, n_samples=size)
        temp_pd = pd.DataFrame({"x1": temp_arr[0].transpose()[0], \
                                "x2": temp_arr[0].transpose()[1], \
                                "x3": temp_arr[0].transpose()[2], \
                                "x4": temp_arr[0].transpose()[3], \
                                "x5": temp_arr[0].transpose()[4], \
                                "y_reg": temp_arr[1]})

        med_point = temp_pd.y_reg.median()

        temp_pd = temp_pd.assign(y=temp_pd[["y_reg"]].apply(lambda x: 1 if (x[0] > med_point) else 0, axis=1))

        df.x24 = temp_pd.x1
        df.x25 = temp_pd.x2
        df.x26 = temp_pd.x3
        df.x27 = temp_pd.x4
        df.x28 = temp_pd.x5

        df = df.assign(y=temp_pd.y)

        # df = df.assign(y=df[["x3", "x4", "x5"]].apply(lambda x: 1 if ( 5 * x[0] > 0.5 or (x[1] >= 0.5 and x[2] == 1) )else 0, axis=1))
        # print(df.corr())
        self.meta_data = pd.DataFrame({"imp_vars": [["x24", "x25", "x26", "x27", "x28"]] * df.shape[0] , "RGS": ["RGS8"] * df.shape[0]})
        self.data = df
        features_names = df.columns.to_list()
        features_names.remove('y')
        self.features_names = features_names
        return self


    def generate_ds8(self, size=10000):
        #   Generate target correlated to two variable, target is true when both 5*x3 or (x4 and x5) are above 0.5 and = 1 respectively .
        #   All variables are strictly positive and continuous except x5 is categorical
        # features = ["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12"]

        data = self.__generate_cor_vars__lst(samples=size, corr=0, num_vars=len(dataset.features))
        df = pd.DataFrame(data, columns=dataset.features)


        df = pd.DataFrame(minmax_scale(df, axis=0, feature_range=(0, 1)), columns=dataset.features)

        df = df.astype(float)
        temp_arr = make_friedman2( n_samples=size)
        temp_pd = pd.DataFrame({"x1": temp_arr[0].transpose()[0], \
                                "x2": temp_arr[0].transpose()[1], \
                                "x3": temp_arr[0].transpose()[2], \
                                "x4": temp_arr[0].transpose()[3], \
                                "y_reg": temp_arr[1]})

        med_point = temp_pd.y_reg.median()

        temp_pd = temp_pd.assign(y=temp_pd[["y_reg"]].apply(lambda x: 1 if (x[0] > med_point) else 0, axis=1))

        df.x29 = temp_pd.x1
        df.x30 = temp_pd.x2
        df.x31 = temp_pd.x3
        df.x32 = temp_pd.x4

        df = df.assign(y=temp_pd.y)

        # df = df.assign(y=df[["x3", "x4", "x5"]].apply(lambda x: 1 if ( 5 * x[0] > 0.5 or (x[1] >= 0.5 and x[2] == 1) )else 0, axis=1))
        # print(df.corr())
        self.meta_data = pd.DataFrame({"imp_vars": [["x29", "x30", "x31", "x32"]] * df.shape[0] , "RGS": ["RGS8"] * df.shape[0]})
        self.data = df
        features_names = df.columns.to_list()
        features_names.remove('y')
        self.features_names = features_names
        return self

    #
    def generate_ds9(self, size=10000):
        #   Generate target correlated to two variable, target is true when both 5*x3 or (x4 and x5) are above 0.5 and = 1 respectively .
        #   All variables are strictly positive and continuous except x5 is categorical
        # features = ["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12"]

        data = self.__generate_cor_vars__lst(samples=size, corr=0, num_vars=len(dataset.features))
        df = pd.DataFrame(data, columns=dataset.features)

        df = pd.DataFrame(minmax_scale(df, axis=0, feature_range=(0, 1)), columns=dataset.features)

        df = df.astype(float)
        temp_arr = make_friedman3(n_samples=size)
        temp_pd = pd.DataFrame({"x1": temp_arr[0].transpose()[0], \
                                "x2": temp_arr[0].transpose()[1], \
                                "x3": temp_arr[0].transpose()[2], \
                                "x4": temp_arr[0].transpose()[3], \
                                "y_reg": temp_arr[1]})

        med_point = temp_pd.y_reg.median()

        temp_pd = temp_pd.assign(y=temp_pd[["y_reg"]].apply(lambda x: 1 if (x[0] > med_point) else 0, axis=1))

        df.x33 = temp_pd.x1
        df.x34 = temp_pd.x2
        df.x35 = temp_pd.x3
        df.x36 = temp_pd.x4

        df = df.assign(y=temp_pd.y)

        # df = df.assign(y=df[["x3", "x4", "x5"]].apply(lambda x: 1 if ( 5 * x[0] > 0.5 or (x[1] >= 0.5 and x[2] == 1) )else 0, axis=1))
        # print(df.corr())
        self.meta_data = pd.DataFrame(
            {"imp_vars": [["x33", "x34", "x35", "x36"]] * df.shape[0], "RGS": ["RGS8"] * df.shape[0]})
        self.data = df
        features_names = df.columns.to_list()
        features_names.remove('y')
        self.features_names = features_names
        return self

    def generate_ds10(self, size=10000):
        """
        Based on sklearn.datasets.make_classification. Based on 4 features.
        This initially creates clusters of points normally distributed (std=1) about vertices of an
        n_informative-dimensional hypercube with sides of length 2*class_sep and assigns an equal
        number of clusters to each class. It introduces interdependence between these features and
        adds various types of further noise to the data.

        Without shuffling, X horizontally stacks features in the following order: the primary n_informative features,
         followed by n_redundant linear combinations of the informative features, followed by n_repeated duplicates,
          drawn randomly with replacement from the informative and redundant features. The remaining features are
          filled with random noise. Thus, without shuffling, all useful features are contained in the
          columns X[:, :n_informative + n_redundant + n_repeated].
        """

        data = self.__generate_cor_vars__lst(samples=size, corr=0, num_vars=len(dataset.features))
        df = pd.DataFrame(data, columns=dataset.features)

        df = pd.DataFrame(minmax_scale(df, axis=0, feature_range=(0, 1)), columns=dataset.features)

        temp_arr = make_classification(n_samples=size, n_features=4, n_informative=4, n_redundant=0, n_repeated=0,
                                       hypercube=False, class_sep=5, random_state=100)
        temp_pd = pd.DataFrame({"x1": temp_arr[0].transpose()[0], \
                                "x2": temp_arr[0].transpose()[1], \
                                "x3": temp_arr[0].transpose()[2], \
                                "x4": temp_arr[0].transpose()[3], \
                                "y": temp_arr[1]})

        df.x37 = temp_pd.x1
        df.x38 = temp_pd.x2
        df.x39 = temp_pd.x3
        df.x40 = temp_pd.x4

        df = df.assign(y=temp_pd.y)
        df.y.where(df.y == 1, 0, inplace=True)

        self.meta_data = pd.DataFrame(
            {"imp_vars": [["x37", "x38", "x39", "x40"]] * df.shape[0], "RGS": ["RGS5"] * df.shape[0]})
        self.data = df
        features_names = df.columns.to_list()
        features_names.remove('y')
        self.features_names = features_names
        return self

    def generate_ds11(self, size=10000):
        """
        sklearn.datasets.make_blobs(n_samples=100, n_features=2, *, centers=None, cluster_std=1.0,
         center_box=(- 10.0, 10.0), shuffle=True, random_state=None, return_centers=False)
        Generate isotropic Gaussian blobs for clustering.
        """

        data = self.__generate_cor_vars__lst(samples=size, corr=0, num_vars=len(dataset.features))
        df = pd.DataFrame(data, columns=dataset.features)

        df = pd.DataFrame(minmax_scale(df, axis=0, feature_range=(0, 1)), columns=dataset.features)

        temp_arr = make_blobs(n_samples=size,  n_features=6, centers=2, cluster_std=8, random_state=100)
        temp_pd = pd.DataFrame({"x1": temp_arr[0].transpose()[0], \
                                "x2": temp_arr[0].transpose()[1], \
                                "x3": temp_arr[0].transpose()[2], \
                                "x4": temp_arr[0].transpose()[3], \
                                "x5": temp_arr[0].transpose()[4], \
                                "x6": temp_arr[0].transpose()[5], \
                                "y": temp_arr[1]})

        df.x41 = temp_pd.x1
        df.x42 = temp_pd.x2
        df.x43 = temp_pd.x3
        df.x44 = temp_pd.x4
        df.x45 = temp_pd.x5
        df.x46 = temp_pd.x6
        df = df.assign(y=temp_pd.y)
        # df.y.where(df.y == 1, 0, inplace=True)

        self.meta_data = pd.DataFrame(
            {"imp_vars": [["x41", "x42", "x43", "x44", "45", "46"]] * df.shape[0], "RGS": ["RGS5"] * df.shape[0]})
        self.data = df
        features_names = df.columns.to_list()
        features_names.remove('y')
        self.features_names = features_names
        return self


    def generate_ds12(self, size=10000):
        """
        Make two interleaving half circles.
        A simple toy dataset to visualize clustering and classification algorithms. Read more in the User Guide.
        """

        data = self.__generate_cor_vars__lst(samples=size, corr=0, num_vars=len(dataset.features))
        df = pd.DataFrame(data, columns=dataset.features)

        df = pd.DataFrame(minmax_scale(df, axis=0, feature_range=(0, 1)), columns=dataset.features)

        temp_arr = make_moons(n_samples=size,   random_state=100)
        temp_pd = pd.DataFrame({"x1": temp_arr[0].transpose()[0], \
                                "x2": temp_arr[0].transpose()[1], \
                                "y": temp_arr[1]})

        df.x47 = temp_pd.x1
        df.x48 = temp_pd.x2

        df = df.assign(y=temp_pd.y)
        # df.y.where(df.y == 1, 0, inplace=True)

        self.meta_data = pd.DataFrame(
            {"imp_vars": [[ "x47", "x48"]] * df.shape[0], "RGS": ["RGS5"] * df.shape[0]})
        self.data = df
        features_names = df.columns.to_list()
        features_names.remove('y')
        self.features_names = features_names
        return self

    # def generate_ds5(self, size=10000):
    #     #   Generate target correlated to two variable, target is true when  x5 = 1  .
    #     #   All variables are strictly positive and continuous except x5 is categorical
    #     # features = ["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12"]
    #
    #     data = self.__generate_cor_vars__lst(samples=size, corr=0, num_vars=len(dataset.features))
    #     df = pd.DataFrame(data, columns=dataset.features)
    #
    #     df = pd.DataFrame(minmax_scale(df, axis=0, feature_range=(0, 1)), columns=dataset.features)
    #     df["x7"] = np.random.choice([1, 2], size)
    #     df = df.astype(float)
    #     df.x7 = df.x7 * 20
    #     mid_point =  df.x7.mean()
    #     df = df.assign(y=df[["x7"]].apply(lambda x: 1 if (x[0] > mid_point) else 0, axis=1))
    #     # print(df.corr())
    #     self.meta_data = pd.DataFrame({"imp_vars": [[ "x7"]] * df.shape[0] , "RGS": ["RGS5"] * df.shape[0]})
    #     self.data = df
    #     features_names = df.columns.to_list()
    #     features_names.remove('y')
    #     self.features_names = features_names
    #     return self

    # def generate_ds6(self, size=10000):
    #     #   Generate target correlated to two variable, target is true when both x4, x5 are above 0.5 and = 1 respectively .
    #     #   All variables are strictly positive and continuous except x5 is categorical
    #     # features = ["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12"]
    #
    #     data = self.__generate_cor_vars__lst(samples=size, corr=0, num_vars=len(dataset.features))
    #     df = pd.DataFrame(data, columns=dataset.features)
    #
    #     df["x9"] = np.random.choice([1, 2], size)
    #
    #     df = pd.DataFrame(minmax_scale(df, axis=0, feature_range=(0, 1)), columns=dataset.features)
    #     df = df.astype(float)
    #     df[["x8", "x9"]] = df[["x8", "x9"]] * 20
    #
    #     mid_point = df[["x8", "x8"]].apply(lambda x: (x[0]  + x[1] )  , axis=1).median()
    #     df = df.assign(y=df[["x8", "x9"]].apply(lambda x: 1 if (x[0]  + x[1] ) > mid_point else 0, axis=1))
    #     # print(df.corr())
    #     self.meta_data = pd.DataFrame({"imp_vars": [["x8", "x9"]] * df.shape[0] , "RGS": ["RGS6"] * df.shape[0]})
    #     self.data = df
    #     features_names = df.columns.to_list()
    #     features_names.remove('y')
    #     self.features_names = features_names
    #     return self

    # def generate_ds7(self, size=10000):
    #     #   Generate target correlated to two variable, target is true when both x4, x5 are above 0.5 and = 1 respectively .
    #     #   All variables are strictly positive and continuous except x5 is categorical
    #     # features = ["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12"]
    #
    #     data = self.__generate_cor_vars__lst(samples=size, corr=0, num_vars=len(dataset.features))
    #     df = pd.DataFrame(data, columns=dataset.features)
    #
    #     df = pd.DataFrame(minmax_scale(df, axis=0, feature_range=(0, 1)), columns=dataset.features)
    #     df["x11"] = np.random.choice([1, 2], size)
    #
    #     df = df.astype(float)
    #     df[["x10", "x11"]] = df[["x10", "x11"]] * 20
    #     mid_point = (df.x10 * df.x11).mean()
    #
    #     df = df.assign(y=df[["x10", "x11"]].apply(lambda x: 1 if (x[0]  * x[1] ) > mid_point else 0, axis=1))
    #     # print(df.corr())
    #     self.meta_data = pd.DataFrame({"imp_vars": [["x10", "x11"]] * df.shape[0] , "RGS": ["RGS7"] * df.shape[0]})
    #     self.data = df
    #     features_names = df.columns.to_list()
    #     features_names.remove('y')
    #     self.features_names = features_names
    #     return self




    # def generate_ds8(self, size=10000):
    #     #   Generate target correlated to two variable, target is true when both 5*x3 or (x4 and x5) are above 0.5 and = 1 respectively .
    #     #   All variables are strictly positive and continuous except x5 is categorical
    #     # features = ["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12"]
    #
    #     data = self.__generate_cor_vars__lst(samples=size, corr=0, num_vars=len(dataset.features))
    #     df = pd.DataFrame(data, columns=dataset.features)
    #
    #
    #     df = pd.DataFrame(minmax_scale(df, axis=0, feature_range=(0, 1)), columns=dataset.features)
    #
    #     df["x14"] = np.random.choice([1, 2], size)
    #
    #     df = df.astype(float)
    #     df[["x12", "x13", "x14"]] = df[["x12", "x13", "x14"]] * 20
    #     midpoint = df[["x12", "x13", "x14"]].apply(      lambda x:       (  (5 * x[0])  + x[1] ) * ( x[2] ) , axis=1).median()
    #     df = df.assign(y=df[["x12", "x13", "x14"]].apply(lambda x: 1 if ( ( (5 * x[0])  + x[1] ) * ( x[2] ) >= midpoint )else 0, axis=1))
    #     # df = df.assign(y=df[["x3", "x4", "x5"]].apply(lambda x: 1 if ( 5 * x[0] > 0.5 or (x[1] >= 0.5 and x[2] == 1) )else 0, axis=1))
    #     # print(df.corr())
    #     self.meta_data = pd.DataFrame({"imp_vars": [["x12", "x13", "x14"]] * df.shape[0] , "RGS": ["RGS8"] * df.shape[0]})
    #     self.data = df
    #     features_names = df.columns.to_list()
    #     features_names.remove('y')
    #     self.features_names = features_names
    #     return self



    # def generate_ds9(self, size=10000):
    #     #   Generate target correlated to two variable, target is true when both  x2^2 or 5*x3 or (x4 and x5) are above 0.5 and = 1 respectively .
    #     #   All variables are strictly positive and continuous except x5 is categorical
    #     # features = ["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12"]
    #
    #     data = self.__generate_cor_vars__lst(samples=size, corr=0, num_vars=len(dataset.features))
    #     df = pd.DataFrame(data, columns=dataset.features)
    #
    #     df["x18"] = np.random.choice([1, 2], size)
    #
    #     df = pd.DataFrame(minmax_scale(df, axis=0, feature_range=(0, 1)), columns=dataset.features)
    #     df = df.astype(float)
    #
    #     df[["x15", "x16", "x17", "x18"]] = df[["x15", "x16", "x17", "x18"]] * 20
    #     mid_point = df[["x15", "x16", "x17", "x18"]].apply(lambda x:  (5 * x[0]) + (2 * x[1]) + (x[2] * x[3]), axis=1).median()
    #     df = df.assign(y=df[["x15", "x16", "x17", "x18"]].apply(lambda x: 1 if (( (5 * x[0]) + (2 * x[1]) + (x[2] * x[3]) ) > mid_point )else 0, axis=1))
    #     # print(df.corr())
    #     self.meta_data = pd.DataFrame({"imp_vars": [["x15", "x16", "x17", "x18"]] * df.shape[0] , "RGS": ["RGS9"] * df.shape[0]})
    #     self.data = df
    #     features_names = df.columns.to_list()
    #     features_names.remove('y')
    #     self.features_names = features_names
    #     return self

    # def generate_ds10(self, size=10000):
    #     #   Generate target correlated to two variable, target is true when both  x2^2 or 5*x3 or (x4 and x5) are above 0.5 and = 1 respectively .
    #     #   All variables are strictly positive and continuous except x5 is categorical
    #     # features = ["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12"]
    #
    #     data = self.__generate_cor_vars__lst(samples=size, corr=0, num_vars=len(dataset.features))
    #     df = pd.DataFrame(data, columns=dataset.features)
    #
    #     df["x23"] = np.random.choice([1, 2], size)
    #
    #     df = pd.DataFrame(minmax_scale(df, axis=0, feature_range=(0, 1)), columns=dataset.features)
    #     df = df.astype(float)
    #     df[["x19", "x20", "x21", "x22", "x23"]] = df[["x19", "x20", "x21", "x22", "x23"]] * 20
    #     mid_point = df[["x19", "x20", "x21", "x22", "x23"]].apply(lambda x: ((5*x[0]) + (3*x[1]) +  (2*x[2]) + (x[3] * x[4] )) , axis=1).median()
    #     df = df.assign(y=df[["x19", "x20", "x21", "x22", "x23"]].apply(lambda x: 1 if ( ((5*x[0]) + (3*x[1]) +  (2*x[2]) + (x[3] * x[4] ) > mid_point))  else 0, axis=1))
    #     # print(df.corr())
    #     self.meta_data = pd.DataFrame({"imp_vars": [["x19", "x20", "x21", "x22", "x23"]] * df.shape[0] , "RGS": ["RGS10"] * df.shape[0]})
    #     self.data = df
    #     features_names = df.columns.to_list()
    #     features_names.remove('y')
    #     self.features_names = features_names
    #     return self

    # def generate_ds11(self, size=10000):
    #     #   Generate target correlated to two variable, target is true when both  x2^2 or 5*x3 or (x4 and x5) are above 0.5 and = 1 respectively .
    #     #   All variables are strictly positive and continuous except x5 is categorical
    #     # features = ["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12"]
    #
    #     data = self.__generate_cor_vars__lst(samples=size, corr=0, num_vars=len(dataset.features))
    #     df = pd.DataFrame(data, columns=dataset.features)
    #
    #     #df["x5"] = np.random.choice(2, size)
    #     df = df.astype(float)
    #     df = pd.DataFrame(minmax_scale(df, axis=0, feature_range=(0, 1)), columns=dataset.features)
    #
    #     df[["x24", "x25", "x26", "x27", "x28"]] =  df[["x24", "x25", "x26", "x27", "x28"]]  * 20
    #     mid_point =      df[["x24", "x25", "x26", "x27", "x28"]].apply(lambda x:      (20*x[0]) + (10*x[1]) + (5*x[2]) + (2*x[3]) + x[4], axis=1).median()
    #     df = df.assign(y=df[["x24", "x25", "x26", "x27", "x28"]].apply(lambda x: 1 if (20*x[0]) + (10*x[1]) + (5*x[2]) + (2*x[3]) + x[4] > mid_point else 0, axis=1))
    #
    #
    #     # print(df.corr())
    #     self.meta_data = pd.DataFrame({"imp_vars": [["x24", "x25", "x26", "x27", "x28"]] * df.shape[0] , "RGS": ["RGS11"] * df.shape[0]})
    #     self.data = df
    #     features_names = df.columns.to_list()
    #     features_names.remove('y')
    #     self.features_names = features_names
    #     return self

    def get_data(self):
        return self.data
