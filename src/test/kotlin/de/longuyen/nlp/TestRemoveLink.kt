package de.longuyen.nlp

import org.junit.jupiter.api.Test

class TestRemoveLink {
    @Test
    fun testRemoveLink(){
        val p = RemoveLink()
        println(p.process("I like this websites https://longuyen.de"))
    }
}