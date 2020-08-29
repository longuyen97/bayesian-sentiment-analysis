package de.longuyen.nlp

import org.jsoup.Jsoup

class RemoveHtmlTags : Preprocessor{
    override fun process(input: String): String {
        return Jsoup.parse(input).text();
    }

}