package de.longuyen.nlp

class RemoveMention : Preprocessor {
    override fun process(input: String): String {
        val ret = mutableListOf<String>()
        val tokens = input.split(" ")
        for(token in tokens){
            if(!token.startsWith("@")){
                ret.add(token.trim())
            }
        }
        return ret.joinToString(" ")
    }
}