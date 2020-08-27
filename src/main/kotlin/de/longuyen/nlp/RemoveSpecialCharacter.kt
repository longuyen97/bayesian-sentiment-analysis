package de.longuyen.nlp

class RemoveSpecialCharacter : Preprocessor {
    private val characters = mutableListOf(
            ",",
            ".",
            "?",
            "\"",
            "'",
            "+",
            "-",
            "*",
    )
    override fun process(input: String): String {
        var processed = input
        for (character in characters){
            processed = processed.replace(character, "")
        }
        return processed
    }
}