package de.longuyen

import de.longuyen.nlp.IO
import de.longuyen.nlp.WeirdWhiteSpacePreprocessing

class Pipeline{
    private val io = IO()
    private val preprocessing = mutableListOf(
        WeirdWhiteSpacePreprocessing()
    )
    private

    fun data() : Pair<Array<String>, Array<Int>>{
        val input = io.read()

        return input
    }
}