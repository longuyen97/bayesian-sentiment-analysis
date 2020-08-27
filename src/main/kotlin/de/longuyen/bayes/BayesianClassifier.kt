package de.longuyen.bayes

interface BayesianClassifier<F, T> {
    fun initialize(documents: Array<Array<F>>, targets: Array<T>)
    fun predict(document: Array<F>) : T
    fun predict(documents: Array<Array<F>>) : List<T>
}