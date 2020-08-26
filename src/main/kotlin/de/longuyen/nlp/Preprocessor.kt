package de.longuyen.nlp

interface Preprocessor {
    fun process(input: String) : String
}