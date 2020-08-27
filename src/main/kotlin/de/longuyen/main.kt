package de.longuyen

import de.longuyen.bayes.RandomForrestClassifier

fun main() {
    val pipeline = Pipeline(RandomForrestClassifier())
    pipeline.train()
}