package de.longuyen.nlp

import org.junit.jupiter.api.Test

class TestRemoveSpecialCharacter {
    @Test
    fun testRemoveSpecialCharacter(){
        val p = RemoveSpecialCharacter()
        println(p.process("I like this websites https://longuyen.de very much."))
    }
}