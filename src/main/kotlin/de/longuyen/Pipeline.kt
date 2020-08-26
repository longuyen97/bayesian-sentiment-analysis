package de.longuyen

import de.longuyen.bayes.BayesianClassifier
import de.longuyen.nlp.IO
import de.longuyen.nlp.NGram
import de.longuyen.nlp.Preprocessor

class Pipeline(private val bayesianClassifier: BayesianClassifier<String, Int>){
    private val io = IO()
    private val preprocessors = mutableListOf<Preprocessor>()
    private val ngram = NGram(1)

    fun train(){
        val input = io.read()
        val features: MutableList<String> = input.first.toMutableList()
        val targets: Array<Int> = input.second

        for(i in features.indices){
            for(preprocessor in preprocessors){
                features[i] = preprocessor.process(features[i])
            }
        }

        val nGramFeatures = mutableListOf<Array<String>>()
        val iter= features.iterator()
        while(iter.hasNext()){
            nGramFeatures.add(ngram.analyze(iter.next()))
            iter.remove()
        }

        bayesianClassifier.initialize(nGramFeatures.toTypedArray(), targets)
    }
}