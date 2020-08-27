package de.longuyen.nlp

class LowerCase : Preprocessor {
    override fun process(input: String): String {
        return input.toLowerCase()
    }
}