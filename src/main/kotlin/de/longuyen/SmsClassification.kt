package de.longuyen

import de.longuyen.bayes.NaiveBayesClassifier
import de.longuyen.data.Dataset
import de.longuyen.data.IO
import de.longuyen.metrics.Accuracy
import de.longuyen.nlp.LowerCase
import de.longuyen.nlp.NGram
import de.longuyen.pipeline.Pipeline
import org.apache.lucene.analysis.custom.CustomAnalyzer


fun main() {
    val pipeline = Pipeline(
        IO(Dataset.SMS),
        mutableListOf(
            LowerCase()
        ),
        Tokenizer(
            CustomAnalyzer.builder()
                .withTokenizer("standard")
                .addTokenFilter("lowercase")
                .addTokenFilter("stop")
                .addTokenFilter("porterstem"),
            NGram(2)
        ),
        NaiveBayesClassifier(),
        mutableListOf(
            Accuracy()
        )
    )
    pipeline.train()
}