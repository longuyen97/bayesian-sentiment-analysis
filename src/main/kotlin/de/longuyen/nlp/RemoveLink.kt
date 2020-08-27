package de.longuyen.nlp

class RemoveLink : Preprocessor {
    override fun process(input: String): String {
        val ret = mutableListOf<String>()
        val tokens = input.split(" ")
        for(token in tokens){
            if(!token.startsWith("http") && !token.startsWith("www")){
                ret.add(token.trim())
            }
        }
        return ret.joinToString(" ")
    }
}