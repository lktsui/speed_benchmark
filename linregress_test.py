import pandas as pd
import numpy as np
from load_data import generate_dataset
from sklearn.linear_model import LinearRegression
from timeit import default_timer as timer

def main():
    dataset = generate_dataset()
    signals = {}
    labels = {}
    dataset_nglo = dataset[dataset['Label'] == 1]
    data_type = 'V'
    temperature = 550
    signals['NGLO'] = pd.concat((dataset_nglo['%s0_%d' % (data_type, temperature)],
                                 dataset_nglo['%s1_%d' % (data_type, temperature)],
                                 dataset_nglo['%s2_%d' % (data_type, temperature)],
                                 ), axis=1)

    dataset_nglo['logCH4'] = np.log(dataset_nglo['CH4/ppm'])
    labels['NGLO'] = np.array(dataset_nglo['logCH4'].values)

    dataset_nghi = dataset[dataset['Label'] == 2]
    dataset_nghi['logCH4'] = np.log(dataset_nghi['CH4/ppm'])
    data_type = 'V'
    temperature = 550
    signals['NGHI'] = pd.concat((dataset_nghi['%s0_%d' % (data_type, temperature)],
                                 dataset_nghi['%s1_%d' % (data_type, temperature)],
                                 dataset_nghi['%s2_%d' % (data_type, temperature)],
                                 ), axis=1)
    labels['NGHI'] = np.array(dataset_nghi['logCH4'].values)

    dataset_ch4only = dataset[dataset['Label'] == 0]
    dataset_ch4only['logCH4'] = np.log(dataset_ch4only['CH4/ppm'])
    data_type = 'V'
    temperature = 550
    signals['CH4Only'] = pd.concat((dataset_ch4only['%s0_%d' % (data_type, temperature)],
                                    dataset_ch4only['%s1_%d' % (data_type, temperature)],
                                    dataset_ch4only['%s2_%d' % (data_type, temperature)],
                                    ), axis=1)
    labels['CH4Only'] = np.array(dataset_ch4only['logCH4'].values)

    dataset_ch4nh3 = dataset[dataset['Label'] == 3]
    dataset_ch4nh3['logCH4'] = np.log(dataset_ch4nh3['CH4/ppm'])
    dataset_ch4nh3['logNH3'] = np.log(dataset_ch4nh3['NH3/ppm'])
    data_type = 'V'
    temperature = 550
    signals['CH4NH3'] = pd.concat((dataset_ch4nh3['%s0_%d' % (data_type, temperature)],
                                   dataset_ch4nh3['%s1_%d' % (data_type, temperature)],
                                   dataset_ch4nh3['%s2_%d' % (data_type, temperature)],
                                   ), axis=1)
    labels['CH4NH3_CH4'] = np.array(dataset_ch4nh3['logCH4'].values)
    labels['CH4NH3_NH3'] = np.array(dataset_ch4nh3['logNH3'].values)

    X = signals['CH4NH3']
    y = labels['CH4NH3_CH4']
    model = LinearRegression()

    train_time_array = []
    inference_time_array = []

    dataset_length = len(X)
    for i in range(0, 10):

        start = timer()
        model.fit(X, y)
        end = timer()

        train_time_array.append(end-start)

    for i in range(0,10):

        start = timer()
        model.predict(X)
        end = timer()
        t_delta = end-start
        t_pp = t_delta/dataset_length
        inference_time_array.append(t_pp)

    average_train_time = np.average(np.array(train_time_array))
    average_inference_time = np.average(np.array(inference_time_array))
    print("Linregress: Train time %e %d pts; inference time %e" % (average_train_time, dataset_length,
                                                                   average_inference_time))


if __name__ == '__main__':
    main()
