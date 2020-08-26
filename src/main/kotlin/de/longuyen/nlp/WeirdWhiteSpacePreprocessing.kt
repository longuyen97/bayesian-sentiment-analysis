package de.longuyen.nlp

class WeirdWhiteSpacePreprocessing : Preprocessing  {
    override fun process(input: String): String {
        return input
            .replace("\\s+", " ")
            .replace("\t", " ")
            .trim()
    }
}