# encoding=utf-8

def get_models():
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.svm import LinearSVC,SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.gaussian_process import GaussianProcessClassifier


    LDA = LinearDiscriminantAnalysis()
    kNN = KNeighborsClassifier(n_jobs=-1,n_neighbors=3)
    SVM = SVC(kernel='poly')
    GMM = GaussianProcessClassifier(kernel='poly',n_jobs=-1)
    # NN = MLPClassifier(hidden_layer_sizes=(int(np.sqrt(X_train.shape[1]*6),)))
    models = [LDA,kNN, SVM, GMM]
    return models



if __name__ == '__main__':
    from EML_data.EML_features import load_data
    from model import traditional_model
    # audios, labels = load_data('EML',n_mfcc=20)
    models = get_models()
    for model in models:
        traditional_model.model(model,n_expands = 10,is_p_test=True,is_norm=True)






