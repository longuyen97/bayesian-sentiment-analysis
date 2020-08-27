package de.longuyen

import de.longuyen.bayes.BayesianClassifier
import de.longuyen.nlp.Accuracy
import de.longuyen.nlp.IO
import de.longuyen.nlp.NGram
import de.longuyen.nlp.Preprocessor
import java.io.Serializable
import java.util.*
import java.util.stream.IntStream

class Pipeline(private val bayesianClassifier: BayesianClassifier<String, Int>) : Serializable {
    companion object {
        private const val serialVersionUID: Long = -4270053884763734247
    }

    private val io = IO()
    private val preprocessors = mutableListOf<Preprocessor>()
    private val ngram = NGram(1)
    private val metric = Accuracy<Int>()

    fun train() {
        var start = System.currentTimeMillis()
        val input = io.read()
        val features: MutableList<String> = input.first.toMutableList()
        val targets: Array<Int> = input.second
        println("Reading data took ${System.currentTimeMillis() - start}ms")

        start = System.currentTimeMillis()
        for (preprocessor in preprocessors) {
            IntStream.range(0, features.size).parallel().forEach {
                features[it] = preprocessor.process(features[it])
            }
        }
        println("Preprocessing data took ${System.currentTimeMillis() - start}ms")

        start = System.currentTimeMillis()
        val nGramFeatures = Array(features.size){ arrayOf<String>()}
        IntStream.range(0, features.size).parallel().forEach {
            nGramFeatures[it] = ngram.analyze(features[it])
        }
        features.clear()
        println("Transforming data into NGram took ${System.currentTimeMillis() - start}ms")

        val trainSize = 0.75
        val X = mutableListOf<Array<String>>()
        val Y = mutableListOf<Int>()
        val x = mutableListOf<Array<String>>()
        val y = mutableListOf<Int>()
        val random = Random()
        for(i in nGramFeatures.indices){
            if(random.nextDouble() < trainSize){
                X.add(nGramFeatures[i])
                Y.add(targets[i])
            }else{
                x.add(nGramFeatures[i])
                y.add(targets[i])
            }
        }

        start = System.currentTimeMillis()
        bayesianClassifier.initialize(X.toTypedArray(), Y.toTypedArray())
        println("Training model took ${System.currentTimeMillis() - start}ms")


        start = System.currentTimeMillis()
        val predictionTrain = bayesianClassifier.predict(X.toTypedArray())
        println("Prediction train took ${System.currentTimeMillis() - start}ms")

        start = System.currentTimeMillis()
        val metricTrain = metric.compute(Y.toTypedArray(), predictionTrain.toTypedArray())
        println("Evaluating training prediction took ${System.currentTimeMillis() - start}ms. Accuracy $metricTrain")

        start = System.currentTimeMillis()
        val predictionTest = bayesianClassifier.predict(x.toTypedArray())
        println("Prediction test took ${System.currentTimeMillis() - start}ms")

        start = System.currentTimeMillis()
        val metricTest = metric.compute(y.toTypedArray(), predictionTest.toTypedArray())
        println("Evaluating testing prediction took ${System.currentTimeMillis() - start}ms. Accuracy $metricTest")
    }

    fun predict(input: String) : Int{
        var processed = input
        for (preprocessor in preprocessors) {
            processed = preprocessor.process(processed)
        }
        val ngramed = ngram.analyze(processed)
        return bayesianClassifier.predict(ngramed)
    }
}