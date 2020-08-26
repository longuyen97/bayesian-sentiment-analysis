package de.longuyen.bayes

import java.lang.IllegalArgumentException

class NaiveBayesClassifier<F, T> : BayesianClassifier<F, T> {
    private val labelDistribution = HashMap<T, Int>()
    private val featureDistribution = HashMap<T, HashMap<F, Int>>()
    private var size = 0
    private val labelProbability = mutableMapOf<T, Double>()

    override fun initialize(documents: Array<Array<F>>, labels: Array<T>) {
        if (documents.size != labels.size) {
            throw IllegalArgumentException("Size of features ${documents.size} is not the same of targets ${labels.size}")
        }
        featureDistribution.clear()
        labelDistribution.clear()
        labelProbability.clear()
        size = labels.size

        for (label in labels) {
            if (labelDistribution.containsKey(label)) {
                labelDistribution[label] = labelDistribution[label]!! + 1
            } else {
                labelDistribution[label] = 1
            }
        }

        if (labelDistribution.size != 2) {
            throw IllegalArgumentException("This classifier only accept binary data with two targets.")
        }

        for (key in labelDistribution.keys) {
            featureDistribution[key] = HashMap()
        }

        for (i in documents.indices) {
            val document = documents[i]
            val correspondingFeatureDistribution = featureDistribution[labels[i]]!!
            for (wj in document) {
                if (correspondingFeatureDistribution.containsKey(wj)) {
                    correspondingFeatureDistribution[wj] = correspondingFeatureDistribution[wj]!! + 1
                } else {
                    correspondingFeatureDistribution[wj] = 1
                }
            }
        }

        for (key in labelDistribution.keys) {
            labelProbability[key] = labelDistribution[key]!!.toDouble() / size.toDouble()
        }
    }

    override fun predict(document: Array<F>): T {
    }
}