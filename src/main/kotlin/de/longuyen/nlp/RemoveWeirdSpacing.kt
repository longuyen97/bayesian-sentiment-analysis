package de.longuyen.nlp

class RemoveWeirdSpacing : Preprocessor {
    override fun process(input: String): String {
        return input
                .replace("\\s+", " ")
                .replace("\t", " ")
                .trim()
    }
}