package de.longuyen.bayes

interface BayesianClassifier<F, T> {
    fun initialize(documents: Array<Array<F>>, labels: Array<T>)
    fun predict(document: Array<F>) : T
}