package de.longuyen

import de.longuyen.bayes.NaiveBayesClassifier
import de.longuyen.data.IO
import de.longuyen.pipeline.Pipeline

fun main() {
    val pipeline = Pipeline(IO("data/spam.csv", featureColumn = 1, targetColumn = 0), NaiveBayesClassifier())
    pipeline.train()
}