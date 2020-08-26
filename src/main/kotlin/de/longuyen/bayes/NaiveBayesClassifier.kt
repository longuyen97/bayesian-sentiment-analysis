package de.longuyen.bayes

import java.lang.IllegalArgumentException
import kotlin.math.ln

class NaiveBayesClassifier<F, T>(private val alpha: Double) : BayesianClassifier<F, T> {
    // How often does a label occurs
    private val labelsCount = HashMap<T, Int>()

    // How often does a feature occurs in each label
    private val featuresInLabelsCount = HashMap<T, HashMap<F, Int>>()

    // How many words does a label has
    private val wordsCountInEachLabel = HashMap<T, Int>()

    // How often a word appears
    private val wordsCount = HashMap<F, Int>()

    // How many documents there are
    private var documentSum = 0L

    // How many words
    private var wordSum = 0L

    override fun initialize(documents: Array<Array<F>>, labels: Array<T>) {
        if (documents.size != labels.size) {
            throw IllegalArgumentException("Size of features ${documents.size} is not the same of targets ${labels.size}")
        } else {
            this.clear(documents)
            for (label in labels) {
                if (labelsCount.containsKey(label)) {
                    labelsCount[label] = labelsCount[label]!! + 1
                } else {
                    labelsCount[label] = 1
                }
                if (!featuresInLabelsCount.containsKey(label)) {
                    featuresInLabelsCount[label] = HashMap()
                }
                if (!wordsCountInEachLabel.containsKey(label)) {
                    wordsCountInEachLabel[label] = 0
                }
            }

            if (labelsCount.size != 2) {
                throw IllegalArgumentException("This classifier only accept binary data with two targets.")
            }

            for (i in documents.indices) {
                val document: Array<F> = documents[i]
                val label: T = labels[i]
                val featuresInThisLabelCount: HashMap<F, Int> = featuresInLabelsCount[label]!!
                for (wj: F in document) {
                    if (featuresInThisLabelCount.containsKey(wj)) {
                        featuresInThisLabelCount[wj] = featuresInThisLabelCount[wj]!! + 1
                    } else {
                        featuresInThisLabelCount[wj] = 1
                    }
                    wordsCountInEachLabel[label] = wordsCountInEachLabel[label]!! + 1
                    if (wordsCount.containsKey(wj)) {
                        wordsCount[wj] = wordsCount[wj]!! + 1
                    } else {
                        wordsCount[wj] = 0
                    }
                    wordSum += 1L
                }
            }
        }
    }

    override fun predict(document: Array<F>): T {
        val labels = labelsCount.keys.toList()
        val reversedProbabilities = mutableListOf<Double>()
        for (label in labels) {
            val labelProb = labelsCount[label]!!.toDouble() / documentSum.toDouble()

            var sumWjCi = 0.0
            for (wj in document) {
                val wjCi = (featuresInLabelsCount[label]!!.getOrDefault(wj, 0).toDouble() + alpha) / (wordsCountInEachLabel[label]!!.toDouble() + alpha * document.size.toDouble())
                sumWjCi += ln(wjCi)
            }

            var sumWj = 0.0
            for(wj in document){
                sumWj += ln(wordsCount.getOrDefault(wj, 0).toDouble() / wordSum)
            }

            reversedProbabilities.add(labelProb + sumWjCi - sumWj)
        }

        var max = reversedProbabilities.first()
        var argMax = 0
        for(i in reversedProbabilities.indices){
            if(reversedProbabilities[i] > max){
                max = reversedProbabilities[i]
                argMax = i
            }
        }
        return labels[argMax]
    }

    private fun clear(documents: Array<Array<F>>) {
        documentSum = documents.size.toLong()
        wordSum = 0L
        featuresInLabelsCount.clear()
        labelsCount.clear()
        wordsCount.clear()
        wordsCountInEachLabel.clear()
    }
}