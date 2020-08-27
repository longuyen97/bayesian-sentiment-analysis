package de.longuyen.nlp

class RemoveHashtag : Preprocessor {
    override fun process(input: String): String {
        return input.replace("#", "")
    }
}