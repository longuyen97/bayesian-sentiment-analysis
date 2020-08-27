package de.longuyen.bayes

import java.lang.IllegalArgumentException
import kotlin.math.ln

class NaiveBayesClassifier<F, T>(private val alpha: Double = 0.5) : BayesianClassifier<F, T> {
    // How often does a target occurs
    val tCount = HashMap<T, Int>()

    // How often a feature appears
    val fCount = HashMap<F, Int>()

    // How often does a feature occurs in each target
    val fTCount = HashMap<T, HashMap<F, Int>>()

    // How many features does a class has
    val tFCount = HashMap<T, Int>()

    // How many documents there are
    var dSum = 0L

    // How many features
    var fSum = 0L

    override fun initialize(documents: Array<Array<F>>, targets: Array<T>) {
        if (documents.size != targets.size) {
            throw IllegalArgumentException("Size of features ${documents.size} is not the same of targets ${targets.size}")
        } else {
            this.clear(documents)
            for (target in targets) {
                // Increment count of a target
                if (tCount.containsKey(target)) {
                    tCount[target] = tCount[target]!! + 1
                } else {
                    tCount[target] = 1
                }

                // Map for feature count in a target
                if (!fTCount.containsKey(target)) {
                    fTCount[target] = HashMap()
                }

                // Count of how many features does a target have
                if (!tFCount.containsKey(target)) {
                    tFCount[target] = 0
                }
            }

            for (idx in documents.indices) {
                val document: Array<F> = documents[idx]
                val label: T = targets[idx]
                val fTC: HashMap<F, Int> = fTCount[label]!!
                for (wj: F in document) {

                    // Count how often a feature appears in a target
                    if (fTC.containsKey(wj)) {
                        fTC[wj] = fTC[wj]!! + 1
                    } else {
                        fTC[wj] = 1
                    }

                    // How how often a feature appears generally
                    if (fCount.containsKey(wj)) {
                        fCount[wj] = fCount[wj]!! + 1
                    } else {
                        fCount[wj] = 1
                    }

                    // Count how many features a target have generally
                    tFCount[label] = tFCount[label]!! + 1

                    // How how many features there are generally
                    fSum += 1L
                }
            }
        }
    }

    override fun predict(document: Array<F>): T {
        val targets = tCount.keys.toList()
        val p = mutableListOf<Double>()

        for (t in targets) {
            val pT = ln((tCount.getOrDefault(t, 0).toDouble() + alpha) / dSum.toDouble())

            var sumWjCi = 0.0
            for (wj in document) {
                val wjCi = (fTCount[t]!!.getOrDefault(wj, 0).toDouble() + alpha) / (tFCount.getOrDefault(t, 0) + alpha * document.size.toDouble())
                sumWjCi += ln(wjCi)
            }

            var sumWj = 0.0
            for(wj in document){
                sumWj += ln(fCount.getOrDefault(wj, alpha).toDouble() / fSum)
            }

            p.add(pT + sumWjCi - sumWj)
        }

        var pMax = p.first()
        var argMax = 0
        for(i in p.indices){
            if(p[i] > pMax){
                pMax = p[i]
                argMax = i
            }
        }
        return targets[argMax]
    }

    private fun clear(documents: Array<Array<F>>) {
        dSum = documents.size.toLong()
        fSum = 0L
        fTCount.clear()
        tCount.clear()
        fCount.clear()
        tFCount.clear()
    }
}