# encoding=utf-8
import numpy as np
def load_EML(X,y,test_size=0.28,random_state=293):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test
def standarscaler(X):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(X)
    X_norm = scaler.transform(X)
    return X_norm

def model(model,n_expands = 10,is_p_test=False,is_norm=True):

    from util import output_results
    from EML_data.EML_features import Features

    EML_feats = Features()
    X, y = EML_feats.get_mfcc()
    X_norm = standarscaler(X)
    X_train, X_test, y_train, y_test = load_EML(X_norm, y)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(model)
    output_results.get_results(y_test, y_pred)

    for i in range(n_expands):
        print('n_expand-{0}'.format(i+1))
        cor_feats, cor_mat = EML_feats.get_cor_feats(X, n_expands=i + 1,
                                                     is_p_test=is_p_test,is_norm=is_norm)
        X_train, X_test, y_train, y_test = load_EML(cor_feats, y)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        output_results.get_results(y_test, y_pred)
