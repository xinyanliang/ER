def get_results(y_test,y_pred):
    from sklearn.metrics import accuracy_score, confusion_matrix
    acc = accuracy_score(y_test, y_pred)
    con_matrix = confusion_matrix(y_test, y_pred)
    print('Accuracy: {0}'.format(acc))
    # print('Confusion_matrix: {0}'.format(con_matrix))