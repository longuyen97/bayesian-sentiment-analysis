package de.longuyen.nlp

import org.junit.jupiter.api.Test

class TestLowerCase {
    @Test
    fun testLowerCase(){
        val p = LowerCase()
        println(p.process("I like this websites https://longuyen.de Very Much."))
    }
}