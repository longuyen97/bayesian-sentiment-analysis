package de.longuyen.nlp

import java.io.FileReader

class RemoveStopWords : Preprocessor{
    private val stopWords = mutableListOf<String>()
    init {
        FileReader("data/stopwords.txt").use {
            stopWords.addAll(it.readLines())
        }
    }
    override fun process(input: String): String {
        val ret = mutableListOf<String>()
        val tokens = input.split(" ")
        for(token in tokens){
            var isStopWord = false
            for(stopWord in stopWords) {
                if (token == stopWord) {
                    isStopWord = true
                    break
                }
            }
            if(!isStopWord){
                ret.add(token)
            }
        }
        return ret.joinToString(" ")
    }
}