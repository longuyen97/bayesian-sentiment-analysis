package de.longuyen.nlp

import edu.stanford.nlp.ling.CoreAnnotations.*
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP
import edu.stanford.nlp.util.CoreMap
import java.util.*


class Lemma : Preprocessor {
    private val pipeline: StanfordCoreNLP

    init {
        val props = Properties()
        props["annotators"] = "tokenize, ssplit, pos, lemma"
        this.pipeline = StanfordCoreNLP(props);
    }

    override fun process(input: String): String {
        val lemmas: MutableList<String> = mutableListOf()
        val document: Annotation = Annotation(input)
        pipeline.annotate(document)
        val sentences: List<CoreMap> = document.get(SentencesAnnotation::class.java)
        for (sentence in sentences) {
            // Iterate over all tokens in a sentence
            for (token in sentence.get(TokensAnnotation::class.java)) {
                // Retrieve and add the lemma for each word into the list of lemmas
                lemmas.add(token.get(LemmaAnnotation::class.java))
            }
        }
        return lemmas.joinToString(" ")
    }

}