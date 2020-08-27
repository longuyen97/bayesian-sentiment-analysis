package de.longuyen.nlp

interface Metrics<T> {
    fun compute(a: Array<T>, b: Array<T>) : Double
}