package de.longuyen.bayes

import java.lang.IllegalArgumentException
import java.util.*
import java.util.concurrent.ConcurrentHashMap
import java.util.stream.IntStream
import kotlin.math.ln
import java.io.Serializable

open class NaiveBayesClassifier<F, T>(private val alpha: Double = 0.5) : BayesianClassifier<F, T> , Serializable {
    companion object {
        private const val serialVersionUID: Long = -4270053884763734247
    }

    // How often does a target occurs
    val tCount = ConcurrentHashMap<T, Int>()

    // How often a feature appears
    val fCount = ConcurrentHashMap<F, Int>()

    // How often does a feature occurs in each target
    val fTCount = ConcurrentHashMap<T, ConcurrentHashMap<F, Int>>()

    // How many features does a class has
    val tFCount = ConcurrentHashMap<T, Int>()

    // How many features does a class has
    val counts = ConcurrentHashMap<String, Long>()

    init {
        counts["fSum"] = 0L
        counts["dSum"] = 0L
    }

    override fun initialize(documents: Array<Array<F>>, targets: Array<T>) {
        if (documents.size != targets.size) {
            throw IllegalArgumentException("Size of features ${documents.size} is not the same of targets ${targets.size}")
        } else {
            Arrays.stream(targets).parallel().forEach { target ->
                // Increment count of a target
                if (tCount.containsKey(target)) {
                    tCount[target] = tCount[target]!! + 1
                } else {
                    tCount[target] = 1
                }

                // Map for feature count in a target
                if (!fTCount.containsKey(target)) {
                    fTCount[target] = ConcurrentHashMap()
                }

                // Count of how many features does a target have
                if (!tFCount.containsKey(target)) {
                    tFCount[target] = 0
                }
            }

            IntStream.range(0, documents.size).parallel().forEach { idx ->
                val document: Array<F> = documents[idx]
                val label: T = targets[idx]
                val fTC: ConcurrentHashMap<F, Int> = fTCount[label]!!
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
                    counts["fSum"] = counts["fSum"]!! + 1
                }
                counts["dSum"] = counts["dSum"]!! + 1
            }
        }
    }


    override fun predict(documents: Array<Array<F>>): List<T> {
        val ret = mutableListOf<T>()

        IntStream.range(0, documents.size)
                .forEach {
                    ret.add(tFCount.keys.first())
                }
        IntStream.range(0, documents.size)
                .parallel()
                .forEach {
                    ret[it] = predict(documents[it])
                }
        return ret
    }

    override fun predict(document: Array<F>): T {
        val targets = tCount.keys.toList()
        val p = mutableListOf<Double>()

        for (t in targets) {
            val pT = ln((tCount.getOrDefault(t, 0).toDouble() + alpha) / counts["dSum"]!!.toDouble())

            var sumWjCi = 0.0
            for (wj in document) {
                val wjCi = (fTCount[t]!!.getOrDefault(wj, 0).toDouble() + alpha) / (tFCount.getOrDefault(t, 0) + alpha * document.size.toDouble())
                sumWjCi += ln(wjCi)
            }

            var sumWj = 0.0
            for (wj in document) {
                sumWj += ln(fCount.getOrDefault(wj, alpha).toDouble() / counts["dSum"]!!)
            }

            p.add(pT + sumWjCi - sumWj)
        }

        var pMax = p.first()
        var argMax = 0
        for (i in p.indices) {
            if (p[i] > pMax) {
                pMax = p[i]
                argMax = i
            }
        }
        return targets[argMax]
    }
}