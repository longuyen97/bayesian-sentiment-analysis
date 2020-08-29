package de.longuyen.pipeline

import de.longuyen.bayes.BayesianClassifier
import de.longuyen.data.IO
import de.longuyen.metrics.Accuracy
import de.longuyen.nlp.Preprocessor
import org.apache.lucene.analysis.Analyzer
import org.apache.lucene.analysis.TokenStream
import org.apache.lucene.analysis.custom.CustomAnalyzer
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute
import java.io.FileOutputStream
import java.io.ObjectOutputStream
import java.io.Serializable
import java.util.*
import java.util.stream.IntStream


fun analyze(text: String): Array<String> {
    val analyzer: Analyzer = CustomAnalyzer.builder()
        .withTokenizer("standard")
        .addTokenFilter("lowercase")
        .addTokenFilter("stop")
        .addTokenFilter("porterstem")
        .build()
    val result: MutableList<String> = ArrayList()
    val tokenStream: TokenStream = analyzer.tokenStream("test", text)
    val attr: CharTermAttribute = tokenStream.addAttribute(CharTermAttribute::class.java)
    tokenStream.reset()
    while (tokenStream.incrementToken()) {
        result.add(attr.toString())
    }
    return result.toTypedArray()
}

class Pipeline(private val io: IO, private val bayesianClassifier: BayesianClassifier<String, String>) : Serializable {
    companion object {
        private const val serialVersionUID: Long = -4270053884763734247
    }

    private val preprocessors = mutableListOf<Preprocessor>(
    )
    private val metric = Accuracy<String>()

    fun train() {
        var start = System.currentTimeMillis()
        val input = io.read()
        val features: MutableList<String> = input.first.toMutableList()
        val targets: MutableList<String> = input.second.toMutableList()
        println("Reading data took ${System.currentTimeMillis() - start}ms")

        start = System.currentTimeMillis()
        for (preprocessor in preprocessors) {
            IntStream.range(0, features.size).parallel().forEach {
                features[it] = preprocessor.process(features[it])
            }
        }
        println("Preprocessing data took ${System.currentTimeMillis() - start}ms")


        start = System.currentTimeMillis()
        val tokens = Array(features.size){ arrayOf<String>()}
        IntStream.range(0, features.size).parallel().forEach {
            tokens[it] = analyze(features[it])
        }
        println("Tokenizing data took ${System.currentTimeMillis() - start}ms")

        val trainSize = 75
        val X = mutableListOf<Array<String>>()
        val Y = mutableListOf<String>()
        val x = mutableListOf<Array<String>>()
        val y = mutableListOf<String>()
        val random = Random(42)
        for(i in tokens.indices){
            if(random.nextInt(100) < trainSize){
                X.add(tokens[i])
                Y.add(targets[i])
            }else{
                x.add(tokens[i])
                y.add(targets[i])
            }
        }
        println("Finishing splitting data. Training data has ${X.size} items. Testing data has ${x.size} items")

        start = System.currentTimeMillis()
        bayesianClassifier.learn(X.toTypedArray(), Y.toTypedArray())
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