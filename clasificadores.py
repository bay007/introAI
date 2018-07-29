from sklearn import tree, naive_bayes, svm, discriminant_analysis 


# X= Un set de datos que describen conjuntos de altura, peso, shoe size
# y= Un set de etiquetas que va relacionada con la Variable X
X = [[168, 89, 27], [170, 88, 27], [167, 80, 26], [168, 78, 26], [160, 64, 24], [
    158, 60, 23.5], [160, 60, 25], [162, 66, 23], [158, 59, 23], [161, 65, 24]]
Y = ["H", "H", "H", "H", "M", "M", "M", "M", "M", "M"]


# este sera el objeto para probar los modelos de prediccion
_sample = [184, 70, 25.5]
sample = [_sample, [168, 89, 27],[165, 72, 23]]

# Arbol de desiciones
clf = tree.DecisionTreeClassifier()
cf = clf.fit(X, Y)
prediction = cf.predict(sample)
print(prediction)


# Clasificador pro baeyes
nb = naive_bayes.GaussianNB()
nb.fit(X, Y)
print(nb.predict(sample))


# Clasificador SVC
ssvm = svm.LinearSVC()
ssvm.fit(X, Y)
print(ssvm.predict(sample))

# Clasificador por QDA
qqda = discriminant_analysis.QuadraticDiscriminantAnalysis()
qqda.fit(X, Y)
print(qqda.predict(sample))



## Para ver los distintos tipos de clasificadores y su comportamiento, remitirse a http://scikit-learn.org/stable/_images/sphx_glr_plot_classifier_comparison_001.png