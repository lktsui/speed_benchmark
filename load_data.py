import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def generate_dataset():

    dataset_CH4_0_C2H6 = pd.DataFrame()
    dataset_NG_LO = pd.DataFrame()
    dataset_NG_HI = pd.DataFrame()
    dataset_CH4_NH3 = pd.DataFrame()
    temperatures = (450, 500, 550, 600)
    for temperature in temperatures:

        # Low Ethane Natural Gas Mixture (Label 1)

        current_dataset = pd.read_csv(os.path.join('datasets',
                                                     'Gen5_CSZ_NGLoEthane_%dC.txt'%temperature), sep=',')

        current_dataset = current_dataset[current_dataset['CH4/ppm'] > 0 ]

        if 'CH4/ppm' not in dataset_NG_LO.columns.values.tolist():
            dataset_NG_LO['CH4/ppm'] = current_dataset['CH4/ppm']
            dataset_NG_LO['C2H6/ppm'] = current_dataset['C2H6/ppm']
            dataset_NG_LO['NH3/ppm'] = current_dataset['NH3/ppm']
            dataset_NG_LO['Label'] = 1*np.ones(len(current_dataset['CH4/ppm']))

        dataset_NG_LO['E_M_ratio'] = dataset_NG_LO['C2H6/ppm']/dataset_NG_LO['CH4/ppm']

        dataset_NG_LO['V0_%d'%temperature] = current_dataset['V0']
        dataset_NG_LO['V1_%d'%temperature] = current_dataset['V1']
        dataset_NG_LO['V2_%d'%temperature] = current_dataset['V2']

        # Hi Ethane Natural Gas Mixture (Label 2)
        current_dataset = pd.read_csv(os.path.join('datasets',
                                                     'Gen5_CSZ_NGHiEthane_%dC.txt'%temperature), sep=',')

        current_dataset = current_dataset[current_dataset['CH4/ppm'] > 0 ]

        if 'CH4/ppm' not in dataset_NG_HI.columns.values.tolist():
            dataset_NG_HI['CH4/ppm'] = current_dataset['CH4/ppm']
            dataset_NG_HI['C2H6/ppm'] = current_dataset['C2H6/ppm']
            dataset_NG_HI['NH3/ppm'] = current_dataset['NH3/ppm']
            dataset_NG_HI['Label'] = 2*np.ones(len(current_dataset['CH4/ppm']))

        dataset_NG_HI['E_M_ratio'] = dataset_NG_HI['C2H6/ppm']/dataset_NG_HI['CH4/ppm']

        dataset_NG_HI['V0_%d'%temperature] = current_dataset['V0']
        dataset_NG_HI['V1_%d'%temperature] = current_dataset['V1']
        dataset_NG_HI['V2_%d'%temperature] = current_dataset['V2']

        # Methane + Ammonia Mixture (Label 3)

        current_dataset = pd.read_csv(os.path.join('datasets',
                                                     'Gen5_CSZ_CH4_C2H6_NH3_%dC.txt'%temperature), sep=',')

        current_dataset = current_dataset[current_dataset['NH3/ppm'] > 0]

        if 'CH4/ppm' not in dataset_CH4_NH3.columns.values.tolist():
            dataset_CH4_NH3['CH4/ppm'] = current_dataset['CH4/ppm']
            dataset_CH4_NH3['C2H6/ppm'] = current_dataset['C2H6/ppm']
            dataset_CH4_NH3['NH3/ppm'] = current_dataset['NH3/ppm']
            dataset_CH4_NH3['Label'] = 3*np.ones(len(current_dataset['CH4/ppm']))

        dataset_CH4_NH3['E_M_ratio'] = dataset_CH4_NH3['C2H6/ppm']/dataset_CH4_NH3['CH4/ppm']

        dataset_CH4_NH3['V0_%d'%temperature] = current_dataset['V0']
        dataset_CH4_NH3['V1_%d'%temperature] = current_dataset['V1']
        dataset_CH4_NH3['V2_%d'%temperature] = current_dataset['V2']

        # Methane with 0% C2H6 - Label = 0

        current_dataset = pd.read_csv(os.path.join('datasets',
                                                     'Gen5_CSZ_CH4_C2H6_NH3_%dC.txt'%temperature), sep=',')

        current_dataset = current_dataset[(current_dataset['CH4/ppm'] > 0) &
                                          (current_dataset['C2H6/ppm'] == 0) &
                                          (current_dataset['NH3/ppm'] == 0)

                                          ]

        if 'CH4/ppm' not in dataset_CH4_0_C2H6.columns.values.tolist():
            dataset_CH4_0_C2H6['CH4/ppm'] = current_dataset['CH4/ppm']
            dataset_CH4_0_C2H6['C2H6/ppm'] = current_dataset['C2H6/ppm']
            dataset_CH4_0_C2H6['NH3/ppm'] = current_dataset['NH3/ppm']
            dataset_CH4_0_C2H6['Label'] = 0*np.ones(len(current_dataset['CH4/ppm']))

        dataset_CH4_0_C2H6['E_M_ratio'] = 0

        dataset_CH4_0_C2H6['V0_%d'%temperature] = current_dataset['V0']
        dataset_CH4_0_C2H6['V1_%d'%temperature] = current_dataset['V1']
        dataset_CH4_0_C2H6['V2_%d'%temperature] = current_dataset['V2']



    dataset_NG_LO = dataset_NG_LO[(dataset_NG_LO['E_M_ratio'] >= 0.020) & (dataset_NG_LO['E_M_ratio'] <= 0.05)]

    total_id_dataset = pd.concat([dataset_CH4_0_C2H6, dataset_CH4_NH3, dataset_NG_LO, dataset_NG_HI],
                                 axis=0)

    features = ['V0_450', 'V1_450', 'V2_450']
    x = total_id_dataset.loc[:, features].values
    # Standardizing the features
    x = StandardScaler().fit_transform(x)
    pca = PCA(n_components=3)
    principalComponents = pca.fit_transform(x)
    principalDf_450 = pd.DataFrame(data=principalComponents
                                   , columns=['PC0_450', 'PC1_450', 'PC2_450'])

    features = ['V0_500', 'V1_500', 'V2_500']
    x = total_id_dataset.loc[:, features].values
    # Standardizing the features
    x = StandardScaler().fit_transform(x)
    pca = PCA(n_components=3)
    principalComponents = pca.fit_transform(x)
    principalDf_500 = pd.DataFrame(data=principalComponents
                                   , columns=['PC0_500', 'PC1_500', 'PC2_500'])

    features = ['V0_550', 'V1_550', 'V2_550']
    x = total_id_dataset.loc[:, features].values
    # Standardizing the features
    x = StandardScaler().fit_transform(x)
    pca = PCA(n_components=3)
    principalComponents = pca.fit_transform(x)
    principalDf_550 = pd.DataFrame(data=principalComponents
                                   , columns=['PC0_550', 'PC1_550', 'PC2_550'])

    features = ['V0_600', 'V1_600', 'V2_600']
    x = total_id_dataset.loc[:, features].values
    # Standardizing the features
    x = StandardScaler().fit_transform(x)
    pca = PCA(n_components=3)
    principalComponents = pca.fit_transform(x)
    principalDf_600 = pd.DataFrame(data=principalComponents
                                   , columns=['PC0_600', 'PC1_600', 'PC2_600'])

    total_id_dataset['PC0_450'] = principalDf_450['PC0_450']
    total_id_dataset['PC1_450'] = principalDf_450['PC1_450']
    total_id_dataset['PC2_450'] = principalDf_450['PC2_450']
    total_id_dataset['PC0_500'] = principalDf_500['PC0_500']
    total_id_dataset['PC1_500'] = principalDf_500['PC1_500']
    total_id_dataset['PC2_500'] = principalDf_500['PC2_500']
    total_id_dataset['PC0_550'] = principalDf_550['PC0_550']
    total_id_dataset['PC1_550'] = principalDf_550['PC1_550']
    total_id_dataset['PC2_550'] = principalDf_550['PC2_550']
    total_id_dataset['PC0_600'] = principalDf_600['PC0_600']
    total_id_dataset['PC1_600'] = principalDf_600['PC1_600']
    total_id_dataset['PC2_600'] = principalDf_600['PC2_600']

    return total_id_dataset

if __name__ == '__main__':

    generate_dataset()