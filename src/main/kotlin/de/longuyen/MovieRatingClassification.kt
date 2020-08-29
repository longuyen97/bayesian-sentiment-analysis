package de.longuyen

import de.longuyen.nlp.NGram
import org.apache.lucene.analysis.custom.CustomAnalyzer
import de.longuyen.bayes.NaiveBayesClassifier
import de.longuyen.data.Dataset
import de.longuyen.data.IO
import de.longuyen.metrics.Accuracy
import de.longuyen.nlp.LowerCase
import de.longuyen.nlp.RemoveHtmlTags
import de.longuyen.pipeline.Pipeline


fun main() {
    val pipeline = Pipeline(
        IO(Dataset.IMDB),
        mutableListOf(
            LowerCase(),
            RemoveHtmlTags()
        ),
        Tokenizer(
            CustomAnalyzer.builder()
            .withTokenizer("standard")
            .addTokenFilter("lowercase")
            .addTokenFilter("stop")
            .addTokenFilter("porterstem"),
            NGram(1)
        ),
        NaiveBayesClassifier(),
        mutableListOf(
            Accuracy()
        )
    )
    pipeline.train()
}