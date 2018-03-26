#encoding=utf-8
import numpy as np
from scipy.stats import stats
def data_expand(X,n_expands=2):
    '''
    计算X的扩张 matrix expanion
    Input:
        X: shape=[n_samples,n_features]
        n_expands: 要扩张的阶
    Output:
        X_expand: shape=[n_samples,n_features*n_expands]
    Example
        X = [[5.5 4.2 1.4 0.2]
             [5.0 3.4 1.5 0.2]
             [6.1 2.9 4.7 1.4]
             [6.2 2.2 4.5 1.5]
             [6.3 2.7 4.9 1.8]
             [6.0 2.2 5.0 1.5]]
        X_expand = [[  5.5   30.25   4.2   17.64   1.4    1.96   0.2    0.04]
                    [  5.    25.     3.4   11.56   1.5    2.25   0.2    0.04]
                    [  6.1   37.21   2.9    8.41   4.7   22.09   1.4    1.96]
                    [  6.2   38.44   2.2    4.84   4.5   20.25   1.5    2.25]
                    [  6.3   39.69   2.7    7.29   4.9   24.01   1.8    3.24]
                    [  6.    36.     2.2    4.84   5.    25.     1.5    2.25]]
    '''
    n_samples,n_features = X.shape
    X_expand = np.zeros((n_samples,n_features * n_expands),dtype=np.float)

    for i in range(n_features):
        cur_feature = X[:,i]
        for j in range(n_expands):
            cur_idx = i*n_expands+j
            X_expand[:,cur_idx] = cur_feature ** (j+1)
    return X_expand


def get_couped_cor_mat(X, X_expand=None, n_expands=2, is_p_test=True):
    '''
    计算coupled 系数矩阵
    Coupled attribute analysis on numerical data. IJCAI-2013
    Input:
        X: 原始的数据表示，shape =[n_samples,n_features]
        X_expand: X的扩展矩阵,默认为None,shape=[n_samples,n_features*n_expands],
                  通过data_expand(X,n_expands=2)方法得到.
        n_expands: X的扩展阶数，默认2
        is_p_test: 两个变量计算person相关系数时，是否p检验，默认Ture，进行p检验
    Output:
        coupled_cor: X的coupled 系数矩阵，shape = [n_features*n_expands,n_features*n_expands]
    '''
    # coupled 系数矩阵 [n_features*n_expands,n_features*n_expands]
    if X_expand is None:
        X_expand = data_expand(X, n_expands)

    n_samples, n_features = X.shape
    len_coupled_cor = n_features * n_expands
    coupled_cor = np.zeros((len_coupled_cor, len_coupled_cor))
    coupled_p = np.zeros((len_coupled_cor, len_coupled_cor))

    for i in range(len_coupled_cor):
        for j in range(len_coupled_cor):
            r, p = stats.pearsonr(X_expand[:, i], X_expand[:, j])
            coupled_cor[i, j] = r
            coupled_p[i, j] = p

    if is_p_test:
        coupled_cor[coupled_p >= 0.05] = 0

    return coupled_cor


def getX_coupled_feas(X, coupled_cor_mat, is_norm=True):
    '''
    计算X的coupled特征
    Input：
        X: 要计算coupled特征矩阵的原始数据表示，shape=[n_samples,n_features]
        coupled_cor_mat: coupled 系数矩阵，shape = [n_features*n_expands,n_features*n_expands],
                            通过get_couped_cor_mat(X,X_expand=None,n_expands=2,is_p_test = True)方法获得.
        is_norm: when True, similar_X = (similar_X-mean)/std
    Output:
        X_coupled_feas:coupled features表示矩阵，shape=[n_samples,n_features*n_expands]
    '''
    n_samples, n_features = X.shape
    n_coupled_feats = coupled_cor_mat.shape[1]
    n_expands = (int)(n_coupled_feats / n_features)

    W = np.array([1 / np.math.factorial(x + 1) for x in range(n_expands)] * n_features)  # 阶乘系数
    X_expand = data_expand(X, n_expands)
    X_expand = X_expand * W
    X_coupled_feas = np.matmul(X_expand, coupled_cor_mat.T)

    from sklearn.preprocessing import StandardScaler
    if is_norm:
        scaler = StandardScaler().fit(X_coupled_feas)
        X_coupled_feas = scaler.transform(X_coupled_feas)

    return X_coupled_feas

def test1():
    X = [[5.5,4.2, 1.4, 0.2],
                 [5.0 ,3.4 ,1.5 ,0.2],
                 [6.1 ,2.9 ,4.7 ,1.4],
                 [6.2 ,2.2 ,4.5 ,1.5],
                 [6.3 ,2.7 ,4.9 ,1.8],
                 [6.0 ,2.2 ,5.0 ,1.5]]
    X = np.array(X)
    cor_mat = get_couped_cor_mat(X,n_expands=2)
    cor_feats = getX_coupled_feas(X,coupled_cor_mat=cor_mat,is_norm=False)
    print(cor_feats)

if __name__ == '__main__':
    test1()