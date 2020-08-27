package de.longuyen.bayes

import java.io.Serializable
import java.lang.Integer.min
import java.util.stream.IntStream

class RandomForrestClassifier<F, T>(private val count: Int = 3) : BayesianClassifier<F, T> , Serializable {
    companion object {
        private const val serialVersionUID: Long = -4270053884763734247
    }

    private val forrest = mutableListOf<NaiveBayesClassifier<F, T>>()

    init {
        for(i in 0 until count){
            forrest.add(NaiveBayesClassifier())
        }
    }

    override fun learn(documents: Array<Array<F>>, targets: Array<T>) {
        val portion = 5
        var k = 0
        for(i in 0 until documents.size - 1 step portion){
            val stop = min(documents.size, i + portion)
            forrest[k % count].learn(documents.copyOfRange(i, stop), targets.copyOfRange(i, stop))
            ++k
        }
    }

    override fun predict(document: Array<F>): T{
        val ret = mutableListOf<T>()
        val dummy = forrest.first().tFCount.keys().toList().first()
        for(tree in forrest){
            ret.add(dummy)
        }
        IntStream.range(0, count)
            .parallel()
            .forEach {
                ret[it] = forrest[it].predict(document)
            }
        return ret.groupingBy { it }.eachCount().maxByOrNull { it.value }!!.key
    }

    override fun predict(documents: Array<Array<F>>): List<T> {
        val ret = mutableListOf<T>()
        val dummy = forrest.first().tFCount.keys().toList().first()
        for(i in documents.indices){
            ret.add(dummy)
        }
        IntStream.range(0, documents.size)
            .parallel()
            .forEach {
                ret[it] = this.predict(documents[it])
            }
        return ret
    }
}