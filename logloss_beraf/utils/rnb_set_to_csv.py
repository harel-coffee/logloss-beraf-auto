import pandas
import rpy2.robjects as robjects
import pandas.rpy.common as com


def rnb_set_to_csv(meth_path, save_to_cv=None):
    x = robjects.r.load(meth_path)
    meth = com.load_data(x[0])
    if save_to_cv is not None:
        meth.to_csv(save_to_cv)
    return meth