package de.longuyen

import de.longuyen.bayes.NaiveBayesClassifier

fun main() {
    val pipeline = Pipeline(NaiveBayesClassifier(0.1))
    pipeline.train()
}