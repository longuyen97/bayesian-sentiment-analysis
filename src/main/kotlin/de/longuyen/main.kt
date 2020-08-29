package de.longuyen

import de.longuyen.bayes.NaiveBayesClassifier
import de.longuyen.data.Dataset
import de.longuyen.data.IO
import de.longuyen.metrics.Accuracy
import de.longuyen.nlp.LowerCase
import de.longuyen.nlp.NGram
import de.longuyen.nlp.RemoveHashtag
import de.longuyen.nlp.RemoveLink
import de.longuyen.pipeline.Pipeline
import org.apache.lucene.analysis.Analyzer
import org.apache.lucene.analysis.TokenStream
import org.apache.lucene.analysis.custom.CustomAnalyzer
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute
import java.util.ArrayList


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

    val string = result.joinToString(" ")
    return NGram(2).analyze(string)
}

fun main() {
    val pipeline = Pipeline(
        IO(Dataset.TWITTER),
        mutableListOf(
            LowerCase(),
            RemoveLink()
        ),
        NaiveBayesClassifier(),
        mutableListOf(
            Accuracy()
        )
    )
    pipeline.train()
}