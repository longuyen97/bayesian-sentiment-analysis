package de.longuyen

import de.longuyen.bayes.BayesianClassifier
import de.longuyen.nlp.*
import java.io.FileOutputStream
import java.io.ObjectOutputStream
import java.io.Serializable
import java.util.*
import java.util.stream.IntStream


class Pipeline(private val bayesianClassifier: BayesianClassifier<String, Int>) : Serializable {
    companion object {
        private const val serialVersionUID: Long = -4270053884763734247
    }

    private val io = IO()
    private val preprocessors = mutableListOf(
            LowerCase(),
            Lemma()
    )
    private val ngram = NGram(1)
    private val metric = Accuracy<Int>()

    fun train() {
        var start = System.currentTimeMillis()
        val input = io.read()
        val features: MutableList<String> = input.first.toMutableList()
        val targets: MutableList<Int> = input.second.toMutableList()
        println("Reading data took ${System.currentTimeMillis() - start}ms")

        start = System.currentTimeMillis()
        for (preprocessor in preprocessors) {
            IntStream.range(0, features.size).parallel().forEach {
                features[it] = preprocessor.process(features[it])
            }
        }
        println("Preprocessing data took ${System.currentTimeMillis() - start}ms")

        start = System.currentTimeMillis()
        val xFiltered = mutableListOf<String>()
        val yFiltered= mutableListOf<Int>()
        IntStream.range(0, features.size).forEach {
            if(features[it].trim().split(this.ngram.delimiter).size >=  this.ngram.n){
                xFiltered.add(features[it])
                yFiltered.add(targets[it])
            }
        }
        println("Filtering data took ${System.currentTimeMillis() - start}ms")

        start = System.currentTimeMillis()
        val nGramFeatures = Array(xFiltered.size){ arrayOf<String>()}
        IntStream.range(0, xFiltered.size).parallel().forEach {
            nGramFeatures[it] = ngram.analyze(xFiltered[it])
        }
        println("Transforming data into NGram took ${System.currentTimeMillis() - start}ms")

        val trainSize = 75
        val X = mutableListOf<Array<String>>()
        val Y = mutableListOf<Int>()
        val x = mutableListOf<Array<String>>()
        val y = mutableListOf<Int>()
        val random = Random(42)
        for(i in nGramFeatures.indices){
            if(random.nextInt(100) < trainSize){
                X.add(nGramFeatures[i])
                Y.add(yFiltered[i])
            }else{
                x.add(nGramFeatures[i])
                y.add(yFiltered[i])
            }
        }
        println("Finishing splitting data. Training data has ${X.size} items. Testing data has ${x.size} items")

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

        FileOutputStream("target/model.ser").use { fos ->
            ObjectOutputStream(fos).use { oos ->
                oos.writeObject(bayesianClassifier)
            }
        }
    }
}