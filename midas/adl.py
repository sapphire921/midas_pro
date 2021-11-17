import datetime
import numpy as np
import pandas as pd

from scipy.optimize import least_squares

from .weights import polynomial_weights

from .mix import mix_freq,mix_freq2
from .fit import ssr, jacobian, ssr2, jacobian2

def estimate(y, yl, x, poly='beta'):
    """
    Fit MIDAS model

    Args:
       y (Series): Low-frequency data
       yl (DataFrame): Lags of low-frequency data
       x (DataFrame): High-frequency regressors

    Returns:
        scipy.optimize.OptimizeResult
    """

    weight_method = polynomial_weights(poly)

    xw, w = weight_method.x_weighted(x, weight_method.init_params())

    # First we do OLS to get initial parameters
    # Note: modified to allow for ylag=0
    if yl is not None:
        c = np.linalg.lstsq(np.concatenate([np.ones((len(xw), 1)), xw.reshape((len(xw), 1)), yl], axis=1), y)[0]

        f = lambda v: ssr(v, x.values, y.values, yl.values, weight_method)
        jac = lambda v: jacobian(v, x.values, y.values, yl.values, weight_method)

        opt_res = least_squares(f,
                            np.concatenate([c[0:2], weight_method.init_params(), c[2:]]),
                            jac,
                            xtol=1e-9,
                            ftol=1e-9,
                            max_nfev=5000,
                            verbose=0)
    
    elif yl is None:
        c = np.linalg.lstsq(np.concatenate([np.ones((len(xw), 1)), xw.reshape((len(xw), 1))], axis=1), y)[0]

        f = lambda v: ssr(v, x.values, y.values, yl, weight_method)
        jac = lambda v: jacobian(v, x.values, y.values, yl, weight_method)

        opt_res = least_squares(f,
                            np.concatenate([c[0:2], weight_method.init_params(), c[2:]]),
                            jac,
                            xtol=1e-9,
                            ftol=1e-9,
                            max_nfev=5000,
                            verbose=0)

    return opt_res


def estimate2(y, yl, x1, x2, poly1='beta',poly2='beta'):
    """
    Fit MIDAS model

    Args:
       y (Series): Low-frequency data
       yl (DataFrame): Lags of low-frequency data
       x1 (DataFrame): High-frequency regressors 1
       x2 (DataFrame): High-frequency regressors 2

    Returns:
        scipy.optimize.OptimizeResult
    """

    weight_method1 = polynomial_weights(poly1)
    weight_method2 = polynomial_weights(poly2)

    xw1, w1 = weight_method1.x_weighted(x1, weight_method1.init_params())
    xw2, w2 = weight_method2.x_weighted(x2, weight_method2.init_params())

    # First we do OLS to get initial parameters
    # Note: modified to allow for ylag=0
    if yl is not None:
        c = np.linalg.lstsq(np.concatenate([np.ones((len(xw1), 1)), xw1.reshape((len(xw1), 1)), xw2.reshape((len(xw2), 1)), yl], axis=1), y)[0]
        #beta0,beta1,beta2,lambda

        f = lambda v: ssr2(v, x1.values, x2.values, y.values, yl.values, weight_method1, weight_method2)
        jac = lambda v: jacobian2(v, x1.values, x2.values, y.values, yl.values, weight_method1, weight_method2)

        opt_res = least_squares(f,
                                np.concatenate([c[0:3], weight_method1.init_params(),
                                                weight_method2.init_params(), c[3:]]),
                                jac,
                                xtol=1e-9,
                                ftol=1e-9,
                                max_nfev=5000,
                                verbose=0)
    elif yl is None:
        c = np.linalg.lstsq(np.concatenate([np.ones((len(xw1), 1)), xw1.reshape((len(xw1), 1)), xw2.reshape((len(xw2), 1))], axis=1), y)[0]
        #beta0,beta1,beta2,lambda

        f = lambda v: ssr2(v, x1.values, x2.values, y.values, yl, weight_method1, weight_method2)
        jac = lambda v: jacobian2(v, x1.values, x2.values, y.values, yl, weight_method1, weight_method2)

        opt_res = least_squares(f,
                                np.concatenate([c[0:3], weight_method1.init_params(),
                                                weight_method2.init_params()]),
                                jac,
                                xtol=1e-9,
                                ftol=1e-9,
                                max_nfev=5000,
                                verbose=0)
        

    return opt_res


def forecast(xfc, yfcl, res, poly='beta'):
    """
    Use the results of MIDAS regression to forecast new periods
    """
    weight_method = polynomial_weights(poly)

    a, b, theta1, theta2= res.x[0:4]
    l=res.x[4:]

    xw, w = weight_method.x_weighted(xfc.values, [theta1, theta2])

    yf = a + b * xw
    for i in range(len(l)):
        yf+=l[i]*yfcl.values[:, i]

    return pd.DataFrame(yf, index=xfc.index, columns=['yfh'])


# Modified to allow for 0 ylag
def forecast2(xfc1, xfc2, yfcl, res, poly1='beta',poly2='beta'):
    """
    Use the results of MIDAS regression to forecast new periods
    """
    weight_method1 = polynomial_weights(poly1)
    weight_method2 = polynomial_weights(poly2)

    a, b1, b2, theta1, theta2,theta3, theta4  = res.x[0:7]
    l=res.x[7:]

    xw1, w1 = weight_method1.x_weighted(xfc1.values, [theta1, theta2])
    xw2, w2 = weight_method2.x_weighted(xfc2.values, [theta3, theta4])

    yf = a + b1 * xw1 +b2 * xw2
    for i in range(len(l)):
        yf+=l[i]*yfcl.values[:, i]

    return pd.DataFrame(yf, index=xfc1.index, columns=['yfh'])


def midas_adl(y_in, x_in, start_date, end_date, xlag, ylag, horizon, forecast_horizon=1, poly='beta', method='fixed'):
    methods = {'fixed': fixed_window,
               'rolling': rolling,
               'recursive': recursive}

    return methods[method](y_in, x_in, start_date, end_date, xlag, ylag, horizon, forecast_horizon, poly)


def fixed_window(y_in, x_in, start_date, end_date, xlag, ylag, horizon, forecast_horizon=1, poly='beta'):

    y, yl, x, yf, ylf, xf = mix_freq(y_in, x_in, xlag, ylag, horizon,
                                     start_date=start_date,
                                     end_date=end_date)

    res = estimate(y, yl, x, poly=poly)

    fc = forecast(xf, ylf, res, poly=poly)

    return (rmse(fc.yfh, yf),
            pd.DataFrame({'preds': fc.yfh, 'targets': yf}, index=yf.index))


def rolling(y_in, x_in, start_date, end_date, xlag, ylag, horizon, forecast_horizon=1, poly='beta'):
    """
    Make a series of forecasts using a fixed-size "rolling window" to fit the
    model

    Args:
        y_in (Series): Dependent variable
        x_in (Series): Independent variables
        start_date: Initial start date for window
        window_size: Number of periods in window
        max_horizon: Maximum horizon to forecast

    Returns:
        rmse (float64), predicted and target values (DataFrame)

    """
    preds = []
    targets = []
    dt_index = []
    start_loc = y_in.index.get_loc(start_date)
    window_size = 60
    if end_date is not None:
        end_loc = y_in.index.get_loc(end_date)
        window_size = end_loc - start_loc

    while start_loc + window_size < (len(y_in.index) - forecast_horizon):
        y, yl, x, yf, ylf, xf = mix_freq(y_in, x_in, xlag, ylag, horizon,
                                         start_date=y_in.index[start_loc],
                                         end_date=y_in.index[start_loc + window_size])
        if len(xf) - forecast_horizon <= 0:
            break

        res = estimate(y, yl, x)

        fc = forecast(xf, ylf, res)

        preds.append(fc.iloc[forecast_horizon - 1].values[0])
        targets.append(yf.iloc[forecast_horizon - 1])
        dt_index.append(yf.index[forecast_horizon - 1])

        start_loc += 1

    preds = np.array(preds)
    targets = np.array(targets)

    return (rmse(preds, targets),
            pd.DataFrame({'preds': preds, 'targets': targets}, index=pd.DatetimeIndex(dt_index)))


def recursive(y_in, x_in, start_date, end_date, xlag, ylag, horizon, forecast_horizon=1, poly='beta'):
    preds = []
    targets = []
    dt_index = []

    forecast_start_loc = y_in.index.get_loc(end_date)

    model_end_dates = y_in.index[forecast_start_loc:-forecast_horizon]

    for estimate_end in model_end_dates:
        y, yl, x, yf, ylf, xf = mix_freq(y_in, x_in, xlag, ylag, horizon,
                                         start_date=start_date,
                                         end_date=estimate_end)
        if len(xf) - forecast_horizon <= 0:
            break

        res = estimate(y, yl, x)

        fc = forecast(xf, ylf, res)

        preds.append(fc.iloc[forecast_horizon - 1].values[0])
        targets.append(yf.iloc[forecast_horizon - 1])
        dt_index.append(yf.index[forecast_horizon - 1])

    preds = np.array(preds)
    targets = np.array(targets)

    return (rmse(preds, targets),
            pd.DataFrame({'preds': preds, 'targets': targets}, index=pd.DatetimeIndex(dt_index)))


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())



def midas_adl2(y_in, x1_in, x2_in, start_date, end_date, x1lag, x2lag, ylag, horizon, forecast_horizon=1, poly1='beta',poly2='beta', method='rolling'):
    methods = {
               'fixed': fixed_window2,
               'rolling': rolling2,
               'recursive': recursive2
               }

    return methods[method](y_in, x1_in, x2_in, start_date, end_date, x1lag, x2lag, ylag, horizon, forecast_horizon, poly1,poly2)


def fixed_window2(y_in, x1_in, x2_in, start_date, end_date, x1lag, x2lag, ylag, horizon, forecast_horizon=1, poly1='beta',poly2='beta'):

    y, yl, x1, x2, yf, ylf, x1f, x2f = mix_freq2(y_in, x1_in, x2_in, x1lag, x2lag, ylag, horizon,
                                     start_date=start_date,
                                     end_date=end_date)

    res = estimate2(y, yl, x1,x2,poly1,poly2)

    fc = forecast2(x1f,x2f, ylf, res,poly1,poly2)

    return (rmse(fc.yfh, yf),
            pd.DataFrame({'preds': fc.yfh, 'targets': yf}, index=yf.index))



def rolling2(y_in, x1_in, x2_in, start_date, end_date, x1lag, x2lag, ylag, horizon, forecast_horizon=1, poly1='beta',poly2='beta'):
    """
    Make a series of forecasts using a fixed-size "rolling window" to fit the
    model

    Args:
        y_in (Series): Dependent variable
        x_in (Series): Independent variables
        start_date: Initial start date for window
        window_size: Number of periods in window
        max_horizon: Maximum horizon to forecast

    Returns:
        rmse (float64), predicted and target values (DataFrame)

    """
    preds = []
    targets = []
    dt_index = []
    start_loc = y_in.index.get_loc(start_date)
    window_size = 60
    if end_date is not None:
        end_loc = y_in.index.get_loc(end_date)
        window_size = end_loc - start_loc

    while start_loc + window_size < (len(y_in.index) - forecast_horizon):
        y, yl, x1, x2, yf, ylf, x1f, x2f = mix_freq2(y_in, x1_in, x2_in, x1lag, x2lag, ylag, horizon,
                                         start_date=y_in.index[start_loc],
                                         end_date=y_in.index[start_loc + window_size])
        if len(x1f) - forecast_horizon <= 0:
            break

        res = estimate2(y, yl, x1,x2,poly1,poly2)

        fc = forecast2(x1f,x2f, ylf, res,poly1,poly2)

        preds.append(fc.iloc[forecast_horizon - 1].values[0])
        targets.append(yf.iloc[forecast_horizon - 1])
        dt_index.append(yf.index[forecast_horizon - 1])

        start_loc += 1

    preds = np.array(preds)
    targets = np.array(targets)

    return (rmse(preds, targets),
            pd.DataFrame({'preds': preds, 'targets': targets}, index=pd.DatetimeIndex(dt_index)))


def recursive2(y_in, x1_in, x2_in, start_date, end_date, x1lag, x2lag, ylag, horizon, forecast_horizon=1, poly1='beta',poly2='beta'):
    preds = []
    targets = []
    dt_index = []

    forecast_start_loc = y_in.index.get_loc(end_date)

    model_end_dates = y_in.index[forecast_start_loc:-forecast_horizon]

    for estimate_end in model_end_dates:
        y, yl, x1, x2, yf, ylf, x1f, x2f = mix_freq2(y_in, x1_in, x2_in, x1lag, x2lag, ylag, horizon,
                                         start_date=start_date,
                                         end_date=estimate_end)
        if len(x1f) - forecast_horizon <= 0:
            break

        res = estimate2(y, yl, x1,x2,poly1,poly2)

        fc = forecast2(x1f,x2f, ylf, res,poly1,poly2)

        preds.append(fc.iloc[forecast_horizon - 1].values[0])
        targets.append(yf.iloc[forecast_horizon - 1])
        dt_index.append(yf.index[forecast_horizon - 1])

    preds = np.array(preds)
    targets = np.array(targets)

    return (rmse(preds, targets),
            pd.DataFrame({'preds': preds, 'targets': targets}, index=pd.DatetimeIndex(dt_index)))