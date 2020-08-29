package de.longuyen

import de.longuyen.nlp.NGram
import org.apache.lucene.analysis.Analyzer
import org.apache.lucene.analysis.TokenStream
import org.apache.lucene.analysis.custom.CustomAnalyzer
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute
import java.util.*

class Tokenizer(private val analyzerBuilder: CustomAnalyzer.Builder, private val ngram: NGram){
    fun analyze(text: String): Array<String> {
        val analyzer = analyzerBuilder.build()
        val result: MutableList<String> = ArrayList()
        val tokenStream: TokenStream = analyzer.tokenStream("test", text)
        val attr: CharTermAttribute = tokenStream.addAttribute(CharTermAttribute::class.java)
        tokenStream.reset()
        while (tokenStream.incrementToken()) {
            result.add(attr.toString())
        }

        val string = result.joinToString(" ")
        return ngram.analyze(string)
    }
}