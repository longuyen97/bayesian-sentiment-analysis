package de.longuyen.nlp

import kotlin.test.Test
import kotlin.test.assertEquals

class IOTest {
    @Test
    fun testReadIO(){
        val io = IO()
        val result = io.read()
        assertEquals(result.first.size, result.second.size)
    }
}